"""SPLADE-v3 retrieval + DeBERTa-v3 reranking over kilt-128 for the public QA pool.

Follows the PISCO §4.1 recipe (SPLADE-v3 first stage, DeBERTa-v3 cross-encoder,
top-5 kept); first-stage depth 50 follows BERGEN's retrieve-then-rerank default.
Three resumable stages, each writing under --work_dir:

  encode  one process per GPU; --gpu_rank/--world split the kilt-128 parquet
          shards; each shard -> enc/shard_XXXXX.npz (scipy CSR, float32).
  search  encode all queries, then stream doc shards through the GPU with
          dense query blocks; keeps a running top-K per query -> search.npz.
  rerank  one process per GPU over a query slice; writes rerank_rank{r}.jsonl
          in the `prepare_public_qa.py attach` contract (documents inlined).

Document id is "kilt128:<id>" and text is the kilt-128 `content` field.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

SPLADE = "/data02/quro/hf-cache/models--naver--splade-v3/snapshots/fdfeceb91d7b9de7985b38addd3ba9f53a59a355"
DEBERTA = "/data02/quro/hf-cache/models--naver--trecdl22-crossencoder-debertav3/snapshots/24f6a61d11707432d5780a1d5cf4e3af25cfaddb"
KILT_DIR = "/data02/quro/hf-cache/datasets--dmrau--kilt-128/blobs"


def shards():
    paths = sorted(glob.glob(os.path.join(KILT_DIR, "shard_*.parquet")))
    if len(paths) != 43:
        raise SystemExit(f"expected 43 kilt-128 shards, found {len(paths)}")
    return paths


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_splade(device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SPLADE)
    model = AutoModelForMaskedLM.from_pretrained(SPLADE, torch_dtype=torch.float16).to(device).eval()
    return tok, model


@torch.no_grad()
def splade_encode(texts, tok, model, device, max_len):
    enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
    logits = model(**enc).logits.float()
    vec = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
    return vec.max(dim=1).values  # (B, vocab), mostly zeros


def to_csr(vec):
    vec = vec.cpu()
    nz = vec.nonzero(as_tuple=True)
    return sp.csr_matrix((vec[nz].numpy(), (nz[0].numpy(), nz[1].numpy())), shape=tuple(vec.shape))


def cmd_encode(args):
    device = "cuda"
    tok, model = load_splade(device)
    out = os.path.join(args.work_dir, "enc")
    os.makedirs(out, exist_ok=True)
    mine = [p for i, p in enumerate(shards()) if i % args.world == args.gpu_rank]
    for path in mine:
        name = os.path.basename(path).replace(".parquet", "")
        target = os.path.join(out, name + ".npz")
        if os.path.exists(target):
            log(f"{name} done, skip")
            continue
        texts = pq.read_table(path, columns=["content"])["content"].to_pylist()
        if args.limit:
            texts = texts[: args.limit]
        t0, parts = time.time(), []
        for i in range(0, len(texts), args.batch):
            parts.append(to_csr(splade_encode(texts[i:i + args.batch], tok, model, device, args.doc_max_len)))
        mat = sp.vstack(parts).tocsr().astype(np.float32)
        sp.save_npz(target + ".tmp.npz", mat)
        os.replace(target + ".tmp.npz", target)
        log(f"{name}: {mat.shape[0]} docs, nnz/doc {mat.nnz / mat.shape[0]:.0f}, "
            f"{mat.shape[0] / (time.time() - t0):.0f} doc/s")


def read_queries(queries_dir):
    rows = []
    for split in ("train", "dev"):
        with open(os.path.join(queries_dir, f"{split}.queries.jsonl"), encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def cmd_search(args):
    device = "cuda"
    queries = read_queries(args.queries_dir)
    tok, model = load_splade(device)
    qvec = torch.cat([splade_encode([q["query"] for q in queries[i:i + 512]], tok, model, device, 128).half()
                      for i in range(0, len(queries), 512)])
    del model
    torch.cuda.empty_cache()
    log(f"encoded {len(queries)} queries")
    k, nq = args.depth, len(queries)
    best_s = torch.full((nq, k), -1.0, device=device)
    best_i = torch.full((nq, k), -1, dtype=torch.long, device=device)
    offset = 0
    for path in shards():
        name = os.path.basename(path).replace(".parquet", "")
        mat = sp.load_npz(os.path.join(args.work_dir, "enc", name + ".npz"))
        docs = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr).long(), torch.from_numpy(mat.indices).long(),
                                       torch.from_numpy(mat.data).half(), size=mat.shape).to(device)
        for s in range(0, nq, args.query_block):
            q = qvec[s:s + args.query_block]
            scores = (docs @ q.T).float().T  # (block, docs_in_shard)
            top_s, top_i = scores.topk(min(k, scores.shape[1]), dim=1)
            cat_s = torch.cat([best_s[s:s + len(q)], top_s], 1)
            cat_i = torch.cat([best_i[s:s + len(q)], top_i + offset], 1)
            keep = cat_s.topk(k, dim=1).indices
            best_s[s:s + len(q)] = cat_s.gather(1, keep)
            best_i[s:s + len(q)] = cat_i.gather(1, keep)
        offset += mat.shape[0]
        log(f"searched {name} (global offset {offset})")
        del docs
        torch.cuda.empty_cache()
    np.savez(os.path.join(args.work_dir, "search.npz"), ids=np.array([q["id"] for q in queries]),
             scores=best_s.cpu().numpy(), doc_index=best_i.cpu().numpy(), total_docs=offset)
    log(f"saved top-{k} for {nq} queries over {offset} docs")


def doc_lookup(indices):
    """Map global row indices to (doc_id, text) by reading only the needed rows."""
    wanted, out, offset = np.unique(indices), {}, 0
    for path in shards():
        table = pq.read_table(path, columns=["id", "content"])
        n = table.num_rows
        local = wanted[(wanted >= offset) & (wanted < offset + n)] - offset
        if len(local):
            ids, texts = table["id"].to_pylist(), table["content"].to_pylist()
            for j in local:
                out[int(j + offset)] = (f"kilt128:{ids[j]}", texts[j])
        offset += n
    return out


def cmd_rerank(args):
    from sentence_transformers import CrossEncoder
    data = np.load(os.path.join(args.work_dir, "search.npz"))
    queries = {q["id"]: q for q in read_queries(args.queries_dir)}
    ids = data["ids"]
    mine = np.arange(len(ids))[args.gpu_rank::args.world]
    target = os.path.join(args.work_dir, f"rerank_rank{args.gpu_rank}.jsonl")
    done = set()
    if os.path.exists(target):
        with open(target, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    mine = [i for i in mine if ids[i] not in done]
    docs = doc_lookup(data["doc_index"][mine].ravel())
    model = CrossEncoder(DEBERTA, device="cuda", max_length=512)
    model.model.half()
    t0 = time.time()
    with open(target, "a", encoding="utf-8") as f:
        for n, i in enumerate(mine, 1):
            q = queries[str(ids[i])]
            cand = [docs[int(j)] for j in data["doc_index"][i] if j >= 0]
            scores = model.predict([(q["query"], text) for _, text in cand], batch_size=args.batch,
                                   show_progress_bar=False)
            order = np.argsort(-np.asarray(scores))[: args.keep]
            f.write(json.dumps({"id": q["id"], "query": q["query"],
                                "documents": [{"doc_id": cand[j][0], "text": cand[j][1]} for j in order]},
                               ensure_ascii=False) + "\n")
            if n % 2000 == 0:
                f.flush()
                log(f"rank {args.gpu_rank}: {n}/{len(mine)} queries, {n / (time.time() - t0):.1f} q/s")
    log(f"rank {args.gpu_rank} done: {len(mine)} new queries")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "search", "rerank"):
        s = sub.add_parser(name)
        s.add_argument("--work_dir", default="/data02/quro/data/public_qa_90k_retrieval")
        s.add_argument("--queries_dir", default="/data02/quro/data/public_qa_90k_questions")
        s.add_argument("--gpu_rank", type=int, default=0)
        s.add_argument("--world", type=int, default=1)
    enc = sub.choices["encode"]
    enc.add_argument("--batch", type=int, default=256)
    enc.add_argument("--doc_max_len", type=int, default=256)
    enc.add_argument("--limit", type=int, default=0, help="debug: first N docs per shard")
    srch = sub.choices["search"]
    srch.add_argument("--depth", type=int, default=50)
    srch.add_argument("--query_block", type=int, default=2048)
    rr = sub.choices["rerank"]
    rr.add_argument("--keep", type=int, default=5)
    rr.add_argument("--batch", type=int, default=50)
    args = p.parse_args()
    {"encode": cmd_encode, "search": cmd_search, "rerank": cmd_rerank}[args.cmd](args)


if __name__ == "__main__":
    main()
