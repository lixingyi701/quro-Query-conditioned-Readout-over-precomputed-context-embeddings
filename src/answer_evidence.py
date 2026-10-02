"""Task-output supervision tied to the answer-paragraph localization result.

Training-only oracle roles define teacher views. Student inputs remain D0/all-Z.
This is a controlled SKD baseline, not a demonstrated new method.
"""
from copy import deepcopy
import json
from pathlib import Path

from .metrics import normalize_answer
from .prompt import BuiltPrompt
from .reader_experiment import file_hash


def validate_target_manifest(path, train_file, eval_file, provenance):
    """Accept only completed, compatible, held-out-clean target exports."""
    if not path:
        raise ValueError("--train_target teacher requires --target_manifest")
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (manifest.get("format") != "answer_evidence_skd_v1" or
            manifest.get("complete") is not True or manifest.get("mode") != "export"):
        raise ValueError("target manifest must be a complete export, not an audit/partial run")
    digest = file_hash(train_file)
    variants = [name for name, info in manifest.get("outputs", {}).items()
                if name != "gold" and info.get("sha256") == digest]
    if not variants:
        raise ValueError("training file is not a hashed teacher output of this manifest")
    named = [name for name in variants if
             Path(manifest["outputs"][name]["file"]).name == Path(train_file).name]
    if named:
        variants = named
    if len(variants) != 1:
        raise ValueError("identical export hashes are ambiguous; retain the original export filename")
    if (variants[0] == "answer_skd_matched" and
            manifest.get("target_counts", {}).get("matched_targets_token_different") == 0):
        raise ValueError("matched target arms are token-identical; do not repeat the candidate")
    teacher = manifest.get("provenance", {})
    # Paths may move; layer metadata is irrelevant to sequence targets. Model,
    # tokenizer, source clipping, memory and target truncation must not change.
    required = ("init_source", "init_checkpoint_sha256", "cache_manifest_sha256",
                "cache_metadata", "tokenizer_sha256", "system_prompt", "max_doc_tokens",
                "max_prompt_tokens", "max_docs", "max_answer_len", "hidden")
    for key in required:
        if key not in teacher or key not in provenance or teacher[key] != provenance[key]:
            raise ValueError(f"target teacher/student provenance mismatch: {key}")
    exclusions = manifest.get("args", {}).get("exclude_files", [])
    inputs = manifest.get("input_sha256", {})
    if file_hash(eval_file) not in {inputs[p] for p in exclusions if p in inputs}:
        raise ValueError("evaluation file was not checked for overlap during target export")
    return {"manifest_sha256": file_hash(path), "compatible_variants": variants,
            "target_counts": manifest.get("target_counts", {})}


def roles(row, builder, corpus):
    """Return original retrieval ranks, or an explicit reason for exclusion."""
    if row.get("hop_type") != "bridge":
        return None, "not_bridge"
    ids = row["retrieved_doc_ids"]
    ranks = row.get("gold_ranks", [])
    if (len(ranks) != 2 or len(set(ranks)) != 2 or
            any(not isinstance(i, int) or i < 0 or i >= len(ids) for i in ranks)):
        return None, "invalid_gold_ranks"
    if any(d not in corpus for d in ids):
        raise ValueError(f"missing corpus document for {row['id']}")
    golds = [normalize_answer(a) for a in row["answers"] if normalize_answer(a)]
    found = [any(a in normalize_answer(builder._clip(corpus[ids[r]])) for a in golds) for r in ranks]
    if sum(found) != 1:
        return None, "ambiguous_answer_paragraph"
    a = found.index(True)
    return {"answer_rank": ranks[a], "bridge_rank": ranks[1-a]}, None


def teacher_view(row, builder, corpus, mode, role=None):
    """M=all memory, R=all raw, A/B=only answer/bridge paragraph raw.

    Keep K and retrieval order fixed; raw text is clipped with the SAME builder.
    The only gold-dependent action is selecting a TRAIN teacher view, never text
    interpolation into a prompt or inference-time student document selection.
    """
    if mode not in {"M", "R", "A", "B"}:
        raise ValueError("unknown teacher view")
    if mode in {"A", "B"} and role is None:
        raise ValueError("mixed teacher requires a unique training role")
    raw_rank = None if role is None else role["answer_rank" if mode == "A" else "bridge_rank"]
    parts, memory_ids, previous = [], [], None
    for rank, doc in enumerate(row["retrieved_doc_ids"]):
        kind = "R" if mode == "R" or (mode in {"A", "B"} and rank == raw_rank) else "M"
        if previous is not None and "R" in (previous, kind):
            parts.append("\n\n")
        if kind == "R":
            parts.append(builder._clip(corpus[doc]))
        else:
            parts.append(builder.slot_string(builder.n_mem_tokens))
            memory_ids.append(doc)
        previous = kind
    text = builder._chat(f"Background:\n{''.join(parts)}\n\nQuestion:{row['query']}")
    ids = builder.tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) >= builder.max_prompt_tokens:
        raise ValueError("teacher prompt exceeds cap; never truncate the question")
    positions = [i for i, t in enumerate(ids) if t in builder.mem_token_ids]
    if len(positions) != len(memory_ids) * builder.n_mem_tokens:
        raise ValueError("teacher memory slots differ from selected cache documents")
    return BuiltPrompt(ids, positions), memory_ids


def acceptable(record, max_answer_len):
    # Substring is a screening heuristic, not a proof that the explanation is true.
    return bool(record and record["substring"] == 1 and record["stopped_eos"]
                and 0 < record["target_length"] <= max_answer_len)


def export_variants(row, conditions, max_answer_len):
    """Same questions/latents/golds; only selected target text differs.

    matched R/A share EXACTLY the same eligibility mask. Everything else keeps
    gold CE. R-all is a stronger ordinary SKD control on all acceptable R rows.
    In particular, R-all is not silently restricted to where the new teacher wins.
    """
    clean = deepcopy(row)
    for key in ("teacher_output", "teacher_answer", "target", "target_source"):
        clean.pop(key, None)
    valid_r = acceptable(conditions.get("R"), max_answer_len)
    valid_a = acceptable(conditions.get("A"), max_answer_len)
    common = valid_r and valid_a
    names = {"gold": None, "raw_skd_all": "R" if valid_r else None,
             "raw_skd_matched": "R" if common else None,
             "answer_skd_matched": "A" if common else None}
    outputs = {}
    for name, mode in names.items():
        result = deepcopy(clean)
        if mode:
            result["teacher_output"] = conditions[mode]["pred"]
        outputs[name] = result
    return outputs, {"raw_eligible": valid_r, "matched_eligible": common,
                     "matched_targets_identical": common and conditions["R"]["pred"] == conditions["A"]["pred"],
                     "raw_target_equals_gold": valid_r and conditions["R"]["pred"].strip() == row["answers"][0].strip(),
                     "answer_target_equals_gold": common and conditions["A"]["pred"].strip() == row["answers"][0].strip()}
