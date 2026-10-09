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
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from prepare_public_qa import digest_file

SPLADE = "/data02/quro/hf-cache/models--naver--splade-v3/snapshots/fdfeceb91d7b9de7985b38addd3ba9f53a59a355"
DEBERTA = "/data02/quro/hf-cache/models--naver--trecdl22-crossencoder-debertav3/snapshots/24f6a61d11707432d5780a1d5cf4e3af25cfaddb"
KILT_DIR = "/data02/quro/hf-cache/datasets--dmrau--kilt-128/blobs"


def shards(args):
    paths = sorted(glob.glob(os.path.join(args.kilt_dir, "shard_*.parquet")))
    if len(paths) != 43:
        raise SystemExit(f"expected 43 kilt-128 shards, found {len(paths)}")
    return paths


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_splade(device, checkpoint):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device).eval()
    return tok, model


def splade_encode(texts, tok, model, device, max_len):
    import torch
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits.float()
        vec = torch.log1p(torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
        return vec.max(dim=1).values  # (B, vocab), mostly zeros


def to_csr(vec):
    vec = vec.cpu()
    nz = vec.nonzero(as_tuple=True)
    return sp.csr_matrix((vec[nz].numpy(), (nz[0].numpy(), nz[1].numpy())), shape=tuple(vec.shape))


def cmd_encode(args):
    import pyarrow.parquet as pq
    device = "cuda"
    out = index_directory(args)
    os.makedirs(out, exist_ok=True)
    bind_index(args)
    tok, model = load_splade(device, args.splade_path)
    mine = [p for i, p in enumerate(shards(args)) if i % args.world == args.gpu_rank]
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


def query_paths(args):
    return (args.queries_jsonl or [os.path.join(args.queries_dir, f"{s}.queries.jsonl") for s in ("train", "dev")])


def read_queries(args):
    rows = []
    seen = set()
    for path in query_paths(args):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in seen or not str(row.get("query", "")).strip():
            raise ValueError(f"missing/duplicate retrieval query ID or text: {identifier}")
        row["id"] = identifier
        seen.add(identifier)
    return rows


def index_directory(args):
    return args.index_dir or os.path.join(args.work_dir, "enc")


def bind_metadata(path, protocol):
    """Reject an output produced for another query pool/configuration."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"stale/incompatible output metadata: {path}; use a new work directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rank-specific temp names also permit identical concurrent index writers.
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(protocol, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, path)


def bind_index(args):
    directory = Path(index_directory(args))
    paths = shards(args)
    protocol = {"splade_path": str(Path(args.splade_path).resolve()),
                "doc_max_len": args.doc_max_len, "debug_limit_per_shard": args.limit,
                "corpus_shards": [{"path": str(Path(p).resolve()), "bytes": os.path.getsize(p)} for p in paths]}
    metadata = directory / "index_metadata.json"
    if not metadata.exists() and list(directory.glob("shard_*.npz")):
        if not args.allow_legacy_index:
            raise ValueError("legacy document index has no provenance metadata; explicitly pass --allow_legacy_index after checking its encoder/corpus")
        log("adopting legacy index under the explicitly supplied encoder/corpus settings")
    bind_metadata(metadata, protocol)
    return protocol


def search_sparse(qvec, matrices, depth, query_block, device="cuda", progress=None):
    """Exact running top-k with CPU query storage and one dense GPU query block.

    GPU query memory is O(query_block * vocabulary), not O(all_questions *
    vocabulary). Running top-k arrays live on CPU; each doc shard is loaded once.
    The CPU path exists to verify the search against exhaustive dot products.
    """
    import torch
    if depth < 1 or query_block < 1 or qvec.shape[0] < 1:
        raise ValueError("positive depth/block and nonempty query vectors required")
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    nq, k = qvec.shape[0], depth
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    offset = 0
    with torch.no_grad():
        for name, mat in matrices:
            if mat.shape[1] != qvec.shape[1] or mat.shape[0] == 0:
                raise ValueError(f"incompatible/empty encoded shard: {name}")
            docs = torch.sparse_csr_tensor(torch.from_numpy(mat.indptr).long(), torch.from_numpy(mat.indices).long(),
                                          torch.from_numpy(mat.data).to(dtype), size=mat.shape).to(device)
            for s in range(0, nq, query_block):
                q = torch.from_numpy(qvec[s:s + query_block].toarray()).to(device=device, dtype=dtype)
                scores = (docs @ q.T).float().T
                top_s, top_i = scores.topk(min(k, scores.shape[1]), dim=1)
                previous_s = torch.from_numpy(best_s[s:s + len(q)]).to(device)
                previous_i = torch.from_numpy(best_i[s:s + len(q)]).to(device)
                cat_s = torch.cat([previous_s, top_s], 1)
                cat_i = torch.cat([previous_i, top_i + offset], 1)
                keep = cat_s.topk(k, dim=1).indices
                best_s[s:s + len(q)] = cat_s.gather(1, keep).cpu().numpy()
                best_i[s:s + len(q)] = cat_i.gather(1, keep).cpu().numpy()
                del q, scores, top_s, top_i, previous_s, previous_i, cat_s, cat_i, keep
            offset += mat.shape[0]
            if progress:
                progress(f"searched {name} (global offset {offset})")
            del docs
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    if np.any(best_i < 0):
        raise ValueError(f"corpus has fewer than {k} documents")
    if not np.isfinite(best_s).all():
        raise ValueError("nonfinite retrieval scores; refuse to rerank invalid candidates")
    return best_s, best_i, offset


def cmd_search(args):
    import torch
    import pyarrow.parquet as pq
    queries = read_queries(args)
    if not queries:
        raise ValueError("no pending questions; skip the search/rerank stage")
    index_protocol = bind_index(args)
    protocol = {"version": "cpu_csr_queries_v1", "index": index_protocol,
                "query_inputs": [{"path": str(Path(p).resolve()), "sha256": digest_file(p)} for p in query_paths(args)],
                "depth": args.depth, "query_max_len": 128, "query_encode_batch": args.query_encode_batch}
    output = Path(args.work_dir) / "search.npz"
    metadata = Path(args.work_dir) / "search_metadata.json"
    if output.exists() and not metadata.exists():
        raise ValueError("existing search has no input fingerprint; use a new work directory")
    bind_metadata(metadata, protocol)
    if output.exists():
        log("matching search already complete, skip")
        return
    qpath = Path(args.work_dir) / "query_vectors.npz"
    if qpath.exists():
        qvec = sp.load_npz(qpath)
        if qvec.shape[0] != len(queries):
            raise ValueError("query vector count differs from bound input")
    else:
        tok, model = load_splade("cuda", args.splade_path)
        parts = []
        for i in range(0, len(queries), args.query_encode_batch):
            parts.append(to_csr(splade_encode([q["query"] for q in queries[i:i + args.query_encode_batch]],
                                             tok, model, "cuda", 128)))
        qvec = sp.vstack(parts).tocsr().astype(np.float32)
        sp.save_npz(str(qpath) + ".tmp.npz", qvec)
        os.replace(str(qpath) + ".tmp.npz", qpath)
        del model, tok, parts
        torch.cuda.empty_cache()
    log(f"encoded {len(queries)} queries into CPU CSR; GPU blocks of {args.query_block}")

    def matrices():
        for path in shards(args):
            name = os.path.basename(path).replace(".parquet", "")
            mat = sp.load_npz(os.path.join(index_directory(args), name + ".npz"))
            expected = pq.ParquetFile(path).metadata.num_rows
            if args.limit:
                expected = min(args.limit, expected)
            if mat.shape[0] != expected:
                raise ValueError(f"{name}: encoded rows differ from the corpus; refuse shifted document IDs")
            yield name, mat
    best_s, best_i, offset = search_sparse(qvec, matrices(), args.depth, args.query_block, progress=log)
    np.savez(str(output) + ".tmp.npz", ids=np.array([q["id"] for q in queries]),
             scores=best_s, doc_index=best_i, total_docs=offset)
    os.replace(str(output) + ".tmp.npz", output)
    log(f"saved top-{args.depth} for {len(queries)} queries over {offset} docs")


def doc_lookup(indices, args):
    """Map global row indices to (doc_id, text) by reading only the needed rows."""
    wanted, out, offset = np.unique(indices), {}, 0
    import pyarrow.parquet as pq
    for path in shards(args):
        table = pq.read_table(path, columns=["id", "content"])
        if args.limit:
            table = table.slice(0, args.limit)
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
    queries = {q["id"]: q for q in read_queries(args)}
    ids = data["ids"]
    search_meta = json.loads((Path(args.work_dir) / "search_metadata.json").read_text())
    if bind_index(args) != search_meta["index"]:
        raise ValueError("rerank document corpus/index differs from the searched corpus")
    actual_inputs = [{"path": str(Path(p).resolve()), "sha256": digest_file(p)} for p in query_paths(args)]
    if search_meta["query_inputs"] != actual_inputs or set(ids.tolist()) != queries.keys():
        raise ValueError("rerank queries differ from search input")
    if args.keep < 1 or args.keep > data["doc_index"].shape[1]:
        raise ValueError("rerank --keep must be within first-stage depth")
    mine = np.arange(len(ids))[args.gpu_rank::args.world]
    target = os.path.join(args.work_dir, f"rerank_rank{args.gpu_rank}.jsonl")
    meta_path = Path(args.work_dir) / f"rerank_rank{args.gpu_rank}.metadata.json"
    if os.path.exists(target) and not meta_path.exists():
        raise ValueError("legacy rerank output has no fingerprint; use a new work directory")
    bind_metadata(meta_path, {"search_sha256": digest_file(Path(args.work_dir) / "search.npz"),
                             "keep": args.keep, "world": args.world, "rank": args.gpu_rank,
                             "reranker": str(Path(args.reranker_path).resolve()),
                             "batch": args.batch, "max_length": 512})
    done = set()
    if os.path.exists(target):
        with open(target, encoding="utf-8") as f:
            completed = [json.loads(line) for line in f if line.strip()]
        done = {r["id"] for r in completed}
        if len(done) != len(completed) or not done <= set(ids[mine].tolist()):
            raise ValueError("rerank output contains duplicate/wrong-rank IDs")
    mine = [i for i in mine if ids[i] not in done]
    if not mine:
        log(f"rank {args.gpu_rank} already complete")
        Path(target).touch(exist_ok=True)
        return
    docs = doc_lookup(data["doc_index"][mine].ravel(), args)
    model = CrossEncoder(args.reranker_path, device="cuda", max_length=512)
    model.model.half()
    t0 = time.time()
    with open(target, "a", encoding="utf-8") as f:
        for n, i in enumerate(mine, 1):
            q = queries[str(ids[i])]
            cand = [docs[int(j)] for j in data["doc_index"][i] if j >= 0]
            scores = model.predict([(q["query"], text) for _, text in cand], batch_size=args.batch,
                                   show_progress_bar=False)
            if not np.isfinite(scores).all():
                raise ValueError(f"nonfinite rerank scores: {q['id']}")
            order = np.argsort(-np.asarray(scores), kind="stable")[: args.keep]
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
        s.add_argument("--queries_jsonl", action="append", help="explicit query JSONL(s), e.g. eval.queries.jsonl")
        s.add_argument("--index_dir", help="reuse a shared enc directory without re-encoding Wikipedia")
        s.add_argument("--kilt_dir", default=KILT_DIR)
        s.add_argument("--splade_path", default=SPLADE)
        s.add_argument("--reranker_path", default=DEBERTA)
        s.add_argument("--doc_max_len", type=int, default=256)
        s.add_argument("--limit", type=int, default=0, help="debug: first N docs per shard; use a separate index")
        s.add_argument("--allow_legacy_index", action="store_true", help="adopt a previously audited index lacking metadata")
        s.add_argument("--gpu_rank", type=int, default=0)
        s.add_argument("--world", type=int, default=1)
    enc = sub.choices["encode"]
    enc.add_argument("--batch", type=int, default=256)
    srch = sub.choices["search"]
    srch.add_argument("--depth", type=int, default=50)
    srch.add_argument("--query_block", type=int, default=128)
    srch.add_argument("--query_encode_batch", type=int, default=64)
    rr = sub.choices["rerank"]
    rr.add_argument("--keep", type=int, default=5)
    rr.add_argument("--batch", type=int, default=50)
    args = p.parse_args()
    if args.world < 1 or not 0 <= args.gpu_rank < args.world:
        p.error("gpu_rank must be within world")
    if args.doc_max_len < 1 or args.limit < 0 or getattr(args, "batch", 1) < 1:
        p.error("positive encoding length/batch and nonnegative debug limit required")
    if args.cmd == "search" and (args.query_block < 1 or args.query_encode_batch < 1 or args.depth < 1):
        p.error("search depth, query block and encoding batch must be positive")
    {"encode": cmd_encode, "search": cmd_search, "rerank": cmd_rerank}[args.cmd](args)


if __name__ == "__main__":
    main()
