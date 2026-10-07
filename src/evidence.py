"""Offline evidence targets and their provenance. Never readout inputs."""
import hashlib
import json

EVIDENCE_INSTRUCTION = "Return the supporting evidence."
ENCODER_RULE = "pisco-decoder-encoder-128-plus-3-right"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def source_digest(row):
    return digest({key: row.get(key) for key in
                   ("id", "query", "retrieved_doc_ids", "supporting_sentences", "gold_ranks")})


def annotate_evidence(row, corpus, tokenizer, manifest, retained_ends=None, cache_digest=None):
    """Keep complete support sentences occurring inside the actual encoder prefix.

    Visibility concerns encoder INPUT, not survival in compressed memories.
    Omit invisible facts instead of imposing impossible text recovery targets.
    """
    if (manifest.get("latent_size") != 8 or manifest.get("doc_max_length") != 128
            or not str(manifest.get("compressor", "")).lower().startswith("pisco")):
        raise ValueError("evidence annotation requires published PISCO 128/8 cache")
    if not tokenizer.is_fast or tokenizer.truncation_side != "right":
        raise ValueError("evidence visibility requires fast, right-truncating tokenizer")
    if "<ENC>" not in tokenizer.get_vocab() or not tokenizer.bos_token or not tokenizer.eos_token:
        raise ValueError("evidence visibility requires PISCO special tokens")
    facts = row.get("supporting_sentences")
    ids = row["retrieved_doc_ids"]
    if not isinstance(facts, list) or any(d not in manifest["documents"] for d in ids):
        raise ValueError(f"{row.get('id')}: missing support facts or cache documents")
    retained_ends = {} if retained_ends is None else retained_ends
    selected, visible, seen = [], [], set()
    for fact in facts:
        rank = fact.get("doc_rank")
        if type(rank) is not int or not 0 <= rank < len(ids):
            raise ValueError(f"{row.get('id')}: invalid support rank")
        if rank not in row.get("gold_ranks", []):
            raise ValueError(f"{row.get('id')}: support sentence outside gold documents")
        doc_id, sentence = ids[rank], str(fact.get("text", "")).strip()
        doc = corpus[doc_id]
        prefix = "<ENC>" + tokenizer.bos_token
        if doc_id not in retained_ends:
            encoded = tokenizer(prefix+doc+tokenizer.eos_token, add_special_tokens=False,
                                truncation=True, max_length=131, return_offsets_mapping=True)
            retained_ends[doc_id] = max((end for start, end in encoded["offset_mapping"]
                                        if end > start), default=0)
        positions, start = [], doc.find(sentence) if sentence else -1
        while start >= 0:
            positions.append(len(prefix)+start+len(sentence))
            start = doc.find(sentence, start+1)
        keep = any(end <= retained_ends[doc_id] for end in positions)
        visible.append(keep)
        if keep and (doc_id, sentence) not in seen:
            selected.append({"doc_id": doc_id, "text": sentence})
            seen.add((doc_id, sentence))
    # Order by document rank, then sentence index. Do not include ranks/titles
    # as extra target labels. They are retained only in offline metadata.
    selected.sort(key=lambda f: (ids.index(f["doc_id"]), next(
        (x.get("sent_id", 0) for x in facts if x["doc_rank"] == ids.index(f["doc_id"])
         and str(x["text"]).strip() == f["text"]), 0)))
    annotation = {"version": 1, "cache_digest": cache_digest or digest(manifest), "source_digest": source_digest(row),
                  "doc_ids": ids, "encoder_rule": ENCODER_RULE, "sentences": selected,
                  "fact_visible": visible, "all_facts_visible": bool(visible) and all(visible),
                  "text": "\n".join(s["text"] for s in selected)}
    annotation["target_digest"] = digest(annotation)
    return {**row, "evidence_annotation": annotation}


def evidence_target(row, doc_ids, cache_digest, require=False):
    annotation = row.get("evidence_annotation")
    if annotation is None:
        if require:
            raise ValueError(f"{row.get('id')}: run prepare_evidence_targets.py first")
        return ""
    payload = {key: value for key, value in annotation.items() if key != "target_digest"}
    original_ids = annotation.get("doc_ids", [])
    original_source = {**row, "retrieved_doc_ids": original_ids}
    if (annotation.get("version") != 1 or annotation.get("cache_digest") != cache_digest
            or annotation.get("source_digest") != source_digest(original_source)
            or original_ids[:len(row["retrieved_doc_ids"])] != row["retrieved_doc_ids"]
            or annotation.get("encoder_rule") != ENCODER_RULE
            or annotation.get("target_digest") != digest(payload)):
        raise ValueError(f"{row.get('id')}: stale/misaligned evidence target")
    sentences = annotation["sentences"]
    if any(s["doc_id"] not in doc_ids for s in sentences):
        return ""  # A document cap must not supervise absent evidence.
    return annotation["text"]
