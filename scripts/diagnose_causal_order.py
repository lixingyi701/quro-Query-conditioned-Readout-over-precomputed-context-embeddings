"""Test-first D0/D2 causal-order probe for PISCO full-cache decoding.

The published PISCO decoder LoRA was trained on D0 (memory before question), so
zero-shot D2 QA is deliberately NOT the test->train gate. The probe asks whether
question-first order creates query-conditioned memory states under exact prompt-
position controls. See docs/CAUSAL_ORDER_D0_D2_TEST_TRAIN_PLAN.md.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import apply_arm, get_config
from src import causal_order as co, metrics, paths
from src.cache import LatentCache
from src.data import read_jsonl
from src.model import build_model
from src.prompt import assemble_inputs


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument("--preset",default="pisco_hotpot"); p.add_argument("--queries")
    p.add_argument("--cache_dir"); p.add_argument("--generator_path"); p.add_argument("--checkpoint")
    p.add_argument("--pairs",type=int,default=64); p.add_argument("--max_docs",type=int)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="auto")
    p.add_argument("--max_answer_tokens",type=int,default=32); p.add_argument("--generate",action="store_true")
    p.add_argument("--max_new_tokens",type=int,default=32); p.add_argument("--out_dir")
    p.add_argument("--gate_min_pairs",type=int,default=32)
    p.add_argument("--gate_min_sensitivity",type=float,default=1e-4)
    p.add_argument("--gate_min_fold",type=float,default=3.0)
    p.add_argument("--gate_min_layer_fraction",type=float,default=.25)
    return p.parse_args()


def device_of(name):
    return torch.device(name if name!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def doc_ids(row,max_docs):
    x=[str(v) for v in row.get("retrieved_doc_ids",[])]; return x[:max_docs] if max_docs else x


def query_positions(tok,builder,query,budget):
    if not getattr(tok,"is_fast",False): raise ValueError("fast tokenizer required")
    text=builder._render(query,budget); prompt=builder.build(query,budget)
    enc=tok(text,add_special_tokens=False,return_offsets_mapping=True,truncation=True,max_length=builder.max_prompt_tokens)
    if list(enc["input_ids"])!=prompt.input_ids: raise ValueError("prompt retokenization mismatch")
    needle="Question:"+query; pos=text.find(needle)
    if pos<0: raise ValueError("cannot locate Question:<query>")
    a=pos+len("Question:"); b=a+len(query); out=[]
    for i,(s,e) in enumerate(enc["offset_mapping"]):
        s,e=int(s),int(e)
        if e<=s: continue
        c=next((j for j in range(s,e) if not text[j].isspace()),s)
        if a<=c<b: out.append(i)
    if not out: raise ValueError("empty query-position set")
    return out


def layout_key(row,model,tok,latent_size,max_docs):
    ids=doc_ids(row,max_docs); budget=len(ids)*latent_size; q=str(row["query"])
    p0=model.prompt_builders["D0"].build(q,budget); p2=model.prompt_builders["D2"].build(q,budget)
    q0=query_positions(tok,model.prompt_builders["D0"],q,budget)
    q2=query_positions(tok,model.prompt_builders["D2"],q,budget)
    return (len(ids),tuple(p0.slot_positions),tuple(q0),tuple(p2.slot_positions),tuple(q2))


def select_pairs(rows,model,tok,cache,n,max_docs,seed):
    buckets=defaultdict(list)
    for row in rows:
        ids=doc_ids(row,max_docs)
        if not ids or not str(row.get("query","")).strip() or any(x not in cache for x in ids): continue
        buckets[layout_key(row,model,tok,cache.metadata.latent_size,max_docs)].append(row)
    rng=random.Random(seed); active=[]
    for _,items in buckets.items():
        rng.shuffle(items)
        if len(items)>=2: active.append(items)
    rng.shuffle(active); pairs=[]
    while active and len(pairs)<n:
        nxt=[]
        for items in active:
            if len(items)>=2 and len(pairs)<n:
                a,b=items.pop(),items.pop()
                if a["query"]!=b["query"]: pairs.append((a,b))
            if len(items)>=2: nxt.append(items)
        active=nxt
    if len(pairs)<n: raise SystemExit(f"only {len(pairs)} exact-position pairs; requested {n}; reduce --pairs, do not relax matching")
    return pairs


def pack(lm,tok,builder,cache,ids,query,target,device):
    dtype=lm.get_input_embeddings().weight.dtype
    z,dm,_=cache.get_many([ids],device=device,dtype=dtype); B,K,M,H=z.shape
    soft=z.reshape(B,K*M,H); mask=dm[:,:,None].expand(B,K,M).reshape(B,K*M); budget=int(mask.sum())
    prompt=builder.build(query,budget); qpos=query_positions(tok,builder,query,budget)
    packed=assemble_inputs(lm.get_input_embeddings(),[prompt],soft,mask,target_ids=[target],pad_token_id=tok.pad_token_id or 0,pad_side="right")
    return prompt,qpos,packed


@torch.no_grad()
def run_trace(lm,tok,builder,cache,ids,query,target,device):
    prompt,qpos,packed=pack(lm,tok,builder,cache,ids,query,target,device)
    with co.capture_block_trace(lm,prompt.slot_positions,qpos) as box: out=lm(**packed,use_cache=False)
    return box["trace"],float(out.loss.detach().float().cpu()),prompt,qpos


@torch.no_grad()
def generate(lm,tok,builder,cache,ids,query,device,max_new):
    dtype=lm.get_input_embeddings().weight.dtype
    z,dm,_=cache.get_many([ids],device=device,dtype=dtype); B,K,M,H=z.shape
    soft=z.reshape(B,K*M,H); mask=dm[:,:,None].expand(B,K,M).reshape(B,K*M); budget=int(mask.sum())
    prompt=builder.build(query,budget)
    packed=assemble_inputs(lm.get_input_embeddings(),[prompt],soft,mask,target_ids=None,pad_token_id=tok.pad_token_id or 0,pad_side="left")
    ids_out=lm.generate(inputs_embeds=packed["inputs_embeds"],attention_mask=packed["attention_mask"],max_new_tokens=max_new,do_sample=False,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
    return tok.decode(ids_out[0].tolist(),skip_special_tokens=True).strip()


def matrix(records,mode,key): return np.asarray([r["modes"][mode][key] for r in records],dtype=float)

def paired(a,b,seed):
    aa=np.nanmean(a,axis=1); bb=np.nanmean(b,axis=1)
    return {"mean_a":float(np.nanmean(aa)),"mean_b":float(np.nanmean(bb)),"paired":co.paired_bootstrap(aa,bb,seed=seed),"layer_mean_a":np.nanmean(a,axis=0).tolist(),"layer_mean_b":np.nanmean(b,axis=0).tolist()}


def main():
    a=args_parser(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg=get_config(a.preset); apply_arm(cfg,"P"); cfg.generator.lora_init="frozen"
    if a.cache_dir: cfg.data.cache_dir=a.cache_dir
    if a.generator_path: cfg.generator.name_or_path=a.generator_path
    if a.max_docs is not None: cfg.data.max_docs=a.max_docs
    cfg.revalidate(); cache=LatentCache(cfg.data.cache_dir); cfg.readout.cache_hidden=cache.metadata.hidden_size
    stack,model=build_model(cfg,cache_hidden=cache.metadata.hidden_size)
    if a.checkpoint: model.load(a.checkpoint,strict=False)
    lm,tok=stack.lm,stack.tokenizer; dev=device_of(a.device); lm.to(dev).eval()
    qfile=a.queries or cfg.data.eval_files.get("dev"); rows=read_jsonl(qfile)
    pairs=select_pairs(rows,model,tok,cache,a.pairs,cfg.data.max_docs,a.seed)
    out=a.out_dir or os.path.join(paths.RUNS_DIR,"causal_order_d0_d2",time.strftime("%Y%m%d-%H%M%S")); os.makedirs(out,exist_ok=False)
    records=[]
    for i,(ra,rb) in enumerate(pairs):
        ida,idb=doc_ids(ra,cfg.data.max_docs),doc_ids(rb,cfg.data.max_docs); qa,qb=str(ra["query"]),str(rb["query"])
        golds=[str(x) for x in (ra.get("answers") or ([ra.get("answer")] if ra.get("answer") else []))]
        target=list(tok(" "+(golds[0] if golds else ""),add_special_tokens=False)["input_ids"][:a.max_answer_tokens]) or [tok.eos_token_id]
        rec={"id":str(ra.get("id",i)),"counterfactual_id":str(rb.get("id","")),"query":qa,"counterfactual_query":qb,"modes":{}}
        for mode in ("D0","D2"):
            b=model.prompt_builders[mode]
            orig,nll,p,qpos=run_trace(lm,tok,b,cache,ida,qa,target,dev)
            qs,_,pq,qq=run_trace(lm,tok,b,cache,ida,qb,target,dev)
            ms,_,pm,qm=run_trace(lm,tok,b,cache,idb,qa,target,dev)
            if p.slot_positions!=pq.slot_positions or qpos!=qq or p.slot_positions!=pm.slot_positions or qpos!=qm: raise RuntimeError("position control failed")
            s={k:v.tolist() for k,v in co.trace_statistics(orig,qs,ms).items()}; s["qa_nll"]=nll
            if a.generate:
                pred=generate(lm,tok,b,cache,ida,qa,dev,a.max_new_tokens); s["prediction"]=pred; s["qa"]=metrics.score(pred,golds or [""])
            rec["modes"][mode]=s
        records.append(rec); print(f"[{i+1}/{len(pairs)}]")
    d0mq,d2mq=matrix(records,"D0","memory_query_cos"),matrix(records,"D2","memory_query_cos")
    d0qm,d2qm=matrix(records,"D0","query_memory_cos"),matrix(records,"D2","query_memory_cos")
    d0dm,d2dm=matrix(records,"D0","delta_memory"),matrix(records,"D2","delta_memory")
    d0dq,d2dq=matrix(records,"D0","delta_query"),matrix(records,"D2","delta_query")
    gate=co.training_gate(d0mq,d2mq,min_pairs=a.gate_min_pairs,min_sensitivity=a.gate_min_sensitivity,min_fold=a.gate_min_fold,min_layer_fraction=a.gate_min_layer_fraction,seed=a.seed)
    summary={"n_pairs":len(records),"training_gate":gate,"primary_memory_query_sensitivity":paired(d2mq,d0mq,a.seed),"complementary_query_memory_sensitivity":paired(d0qm,d2qm,a.seed+1),"memory_update_D2_minus_D0":paired(d2dm,d0dm,a.seed+2),"query_update_D0_minus_D2":paired(d0dq,d2dq,a.seed+3),"mean_nll_D0":float(np.mean([r["modes"]["D0"]["qa_nll"] for r in records])),"mean_nll_D2":float(np.mean([r["modes"]["D2"]["qa_nll"] for r in records])),"qa_is_training_gate":False}
    if a.generate:
        summary["qa_D0"]=metrics.aggregate([r["modes"]["D0"]["qa"] for r in records]); summary["qa_D2"]=metrics.aggregate([r["modes"]["D2"]["qa"] for r in records])
    with open(os.path.join(out,"pairs.jsonl"),"w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with open(os.path.join(out,"summary.json"),"w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    manifest={"preset":a.preset,"queries":qfile,"cache_dir":cfg.data.cache_dir,"decoder":cfg.generator.name_or_path,"checkpoint":a.checkpoint,"pair_rule":"exact D0/D2 memory and query positions","seed":a.seed,"gate_thresholds":gate["thresholds"],"qa_is_training_gate":False}
    with open(os.path.join(out,"manifest.json"),"w",encoding="utf-8") as f: json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps(gate,ensure_ascii=False,indent=2)); print("[done]",out)

if __name__=="__main__": main()
