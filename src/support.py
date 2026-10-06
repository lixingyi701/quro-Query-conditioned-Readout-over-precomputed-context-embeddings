"""Document support targets and a balanced auxiliary loss; never model inputs."""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from .support_metrics import support_scores


def manifest_digest(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def support_source_digest(row):
    return manifest_digest({"id": row.get("id"), "query": str(row.get("query", "")).strip(),
                            "gold_ranks": row.get("gold_ranks"),
                            "supporting_sentences": row.get("supporting_sentences")})


def gold_labels(row, doc_ids):
    ranks = row.get("gold_ranks")
    if ranks is None:
        return None
    if not isinstance(ranks, list) or any(type(i) is not int or i < 0 for i in ranks):
        raise ValueError(f"{row.get('id')}: malformed gold_ranks")
    ranks = set(ranks)
    return [int(i in ranks) for i in range(len(doc_ids))]


def support_targets(row, doc_ids, cache_digest, policy="visible", require=False):
    """Return original labels, their mask, the loss mask, and visibility.

    doc_ids must carry this row's original document order. Query mismatch is an
    evaluation diagnostic against the ORIGINAL question's labels; document
    mismatch has no aligned labels and must be masked by the caller.
    """
    labels = gold_labels(row, doc_ids)
    n = len(doc_ids)
    if labels is None:
        return [0]*n, [False]*n, [False]*n, [None]*n
    annotation = row.get("support_annotation")
    visible = [None]*n
    if annotation is not None:
        if not isinstance(annotation, dict):
            raise ValueError(f"{row.get('id')}: malformed support annotation")
        source_ids = annotation.get("doc_ids", [])
        source_labels = gold_labels(row, source_ids)
        source_visible = annotation.get("visible", [])
        if (annotation.get("version") != 1
                or annotation.get("cache_digest") != cache_digest
                or annotation.get("source_digest") != support_source_digest(row)
                or annotation.get("encoder_rule") != "pisco-decoder-encoder-128-plus-3-right"
                or source_ids[:n] != list(doc_ids) or len(source_ids) < n
                or annotation.get("labels") != source_labels
                or len(source_visible) != len(source_ids)
                or any(v is not None and type(v) is not bool for v in source_visible)
                or any(i >= len(source_ids) for i in row["gold_ranks"])):
            raise ValueError(f"{row.get('id')}: stale/misaligned support annotation")
        visible = source_visible[:n]
    elif require and policy == "visible":
        raise ValueError(f"{row.get('id')}: run annotate_support_visibility.py first")
    mask = [True]*n
    loss_mask = [bool(not y or policy == "original" or v is True)
                 for y, v in zip(labels, visible)]
    return labels, mask, loss_mask, visible


def balanced_support_loss(logits, labels, mask):
    """Per-question mean of positive/negative BCE; skip one-class questions.

    Absent/invisible positives are ignored, never converted into negatives.
    torch.where isolates masked NaNs, including their backward path.
    """
    if logits.shape != labels.shape or logits.shape != mask.shape:
        raise ValueError("support logits/labels/mask must have the same B,K shape")
    mask = mask.bool()
    if bool(((labels != 0) & (labels != 1) & mask).any()):
        raise ValueError("support labels must be binary at supervised positions")
    safe = torch.where(mask, logits, 0.0)
    pos, neg = mask & (labels == 1), mask & (labels == 0)
    np, nn = pos.sum(1), neg.sum(1)
    active = (np > 0) & (nn > 0)
    losses = 0.5 * ((F.softplus(-safe)*pos).sum(1)/np.clamp_min(1)
                    + (F.softplus(safe)*neg).sum(1)/nn.clamp_min(1))
    if bool(active.any()):
        return losses[active].mean(), active.sum()
    return safe.sum()*0.0, active.sum()


def annotate_row(row, corpus, tokenizer, cache_manifest, encoder_length=128,
                 cache_digest=None, retained_end_cache=None):
    """Audit the actual published PISCO decoder-as-encoder input prefix.

    Mirrors third_party/modelling_pisco.py: <ENC><bos>document<eos>,
    add_special_tokens=False, right truncation to doc_max_length+3. No LM loads.
    A retained sentence is an input-visibility proxy, NOT proof it survives Z.
    """
    doc_ids = row["retrieved_doc_ids"]
    labels = gold_labels(row, doc_ids)
    if labels is None or any(i >= len(doc_ids) for i in row["gold_ranks"]):
        raise ValueError(f"{row.get('id')}: complete gold_ranks required")
    if not tokenizer.is_fast or tokenizer.truncation_side != "right":
        raise ValueError("visibility needs a fast tokenizer with right truncation")
    length = cache_manifest.get("doc_max_length") or encoder_length
    if length != 128 or cache_manifest.get("latent_size") != 8:
        raise ValueError("visibility audit currently supports published PISCO 128/8 caches")
    if "<ENC>" not in tokenizer.get_vocab() or not tokenizer.bos_token or not tokenizer.eos_token:
        raise ValueError("use the published PISCO tokenizer including its special tokens")
    facts = row.get("supporting_sentences")
    if not isinstance(facts, list):
        raise ValueError(f"{row.get('id')}: supporting_sentences required")
    visible = []
    for i, (doc_id, label) in enumerate(zip(doc_ids, labels)):
        if not label:
            visible.append(None)
            continue
        document = corpus[doc_id]
        prefix = "<ENC>" + tokenizer.bos_token
        wrapped = prefix + document + tokenizer.eos_token
        if retained_end_cache is not None and doc_id in retained_end_cache:
            retained_end = retained_end_cache[doc_id]
        else:
            encoded = tokenizer(wrapped, add_special_tokens=False, truncation=True,
                                max_length=length+3, return_offsets_mapping=True)
            offsets = encoded["offset_mapping"]
            retained_end = max((end for start, end in offsets if end > start), default=0)
            if retained_end_cache is not None:
                retained_end_cache[doc_id] = retained_end
        support = [f["text"].strip() for f in facts if f["doc_rank"] == i and f["text"].strip()]
        # Repeated sentences can be located at multiple positions; one retained
        # complete occurrence is enough for this document-level visibility proxy.
        occurrences = []
        for sentence in support:
            start = document.find(sentence)
            while start >= 0:
                occurrences.append(len(prefix)+start+len(sentence))
                start = document.find(sentence, start+1)
        visible.append(any(end <= retained_end for end in occurrences) if support else None)
    output = dict(row)
    output["support_annotation"] = {
        "version": 1, "doc_ids": doc_ids, "labels": labels, "visible": visible,
        "cache_digest": cache_digest or manifest_digest(cache_manifest),
        "source_digest": support_source_digest(row),
        "encoder_rule": "pisco-decoder-encoder-128-plus-3-right",
    }
    return output
