"""Test supervision semantics, real decoder scoring, export-to-training contract."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_reader_experiment import tiny_base, sample_batch
from test_reader_reset import script
from src.answer_evidence import roles, teacher_view, export_variants, validate_target_manifest
from src.cache import CacheMetadata, LatentCache, LatentCacheWriter
from src.causal_order import paired_bootstrap
from src.model import QuROModel, TokenEmbeddingQueryEncoder
from src.reader_experiment import ReaderExperiment, ReaderSpec, file_hash, row_key
from src import reader_runtime


def example():
    row = {"id": "x", "query": "Where born ?", "answers": ["Nanjing"],
           "retrieved_doc_ids": ["d0", "d1", "d2"], "gold_ranks": [1, 2], "hop_type": "bridge"}
    corpus = {"d0": "long", "d1": "director one", "d2": "born Nanjing"}
    return row, corpus


def test_teacher_views_preserve_original_prompt_and_retrieval_order():
    base = tiny_base()
    builder = base.prompt_builders["D0"]
    row, corpus = example()
    role, reason = roles(row, builder, corpus)
    assert role == {"answer_rank": 2, "bridge_rank": 1} and reason is None
    for mode, keep, kinds in (("M", row["retrieved_doc_ids"], "MMM"),
                               ("R", [], "RRR"), ("A", ["d0", "d1"], "MMR"),
                               ("B", ["d0", "d2"], "MRM")):
        prompt, memory = teacher_view(row, builder, corpus, mode, role)
        assert memory == keep
        mixed = script("eval_gold_mixed").render_mixed(builder, row["query"],
            [(kind, corpus[doc]) for kind, doc in zip(kinds, row["retrieved_doc_ids"])])
        assert prompt == mixed
        if mode == "M":
            assert prompt == builder.build(row["query"], 24)
        elif mode == "R":
            assert prompt == base.prompt_builders["RG"].build(row["query"], 0,
                [corpus[d] for d in row["retrieved_doc_ids"]])
    # Gold aliases select a training role, never enter the rendered prompt.
    changed = {**row, "answers": ["invented answer absent from all documents"]}
    assert teacher_view(row, builder, corpus, "A", role) == teacher_view(changed, builder, corpus, "A", role)
    builder.max_prompt_tokens = 2
    with pytest.raises(ValueError, match="cap"):
        teacher_view(row, builder, corpus, "A", role)


def test_role_exclusions_and_source_clipping():
    row, corpus = example()
    builder = tiny_base().prompt_builders["D0"]
    assert roles({**row, "hop_type": "comparison"}, builder, corpus)[1] == "not_bridge"
    assert roles({**row, "gold_ranks": [2, 2]}, builder, corpus)[1] == "invalid_gold_ranks"
    assert roles(row, builder, {**corpus, "d1": "Nanjing"})[1] == "ambiguous_answer_paragraph"
    builder.max_doc_tokens = 1  # Answer outside the compressor's view must not label a role.
    assert roles(row, builder, corpus)[1] == "ambiguous_answer_paragraph"


def test_common_mask_and_full_raw_control_do_not_select_only_favorable_rows():
    row, _ = example()
    row["teacher_output"] = "stale teacher"
    before = deepcopy(row)
    good = {"pred": "Nanjing one", "substring": 1., "stopped_eos": True, "target_length": 2}
    variants, flags = export_variants(row, {"R": good, "A": {**good, "substring": 0}}, 48)
    assert row == before
    assert "teacher_output" in variants["raw_skd_all"]
    assert all("teacher_output" not in variants[n] for n in ("gold", "raw_skd_matched", "answer_skd_matched"))
    assert flags["raw_eligible"] and not flags["matched_eligible"]
    for bad in ({**good, "stopped_eos": False}, {**good, "target_length": 49}, {**good, "target_length": 0}):
        assert not export_variants(row, {"R": bad, "A": good}, 48)[1]["matched_eligible"]
    variants, flags = export_variants(row, {"R": good, "A": {**good, "pred": "Nanjing two"}}, 48)
    assert flags["matched_eligible"] and not flags["matched_targets_identical"]
    for variant in variants.values():
        for key in ("id", "query", "answers", "retrieved_doc_ids", "gold_ranks"):
            assert variant[key] == row[key]


def test_real_teacher_gold_nll_and_generation_equal_existing_d0(tmp_path):
    cli = script("build_answer_evidence_targets")
    base = tiny_base()
    reader = ReaderExperiment(base, ReaderSpec("direct-ce", (2,)))
    reader.eval()
    reader.max_answer_len = 48
    batch = sample_batch(base.tok)
    # Existing evaluation and audit must use the same gold token sequence.
    batch["target_ids"] = [[base.tok.stoi[r["answers"][0]], base.tok.eos_token_id] for r in batch["raw"]]
    root = tmp_path/"cache"
    with LatentCacheWriter(root, CacheMetadata("test", 8, 32, "float32", doc_max_length=128)) as writer:
        writer.add("d1", batch["cached_latents"][0, 0], 8)
        writer.add("d2", batch["cached_latents"][0, 1], 8)
    batch["cached_latents"][1, 0] = batch["cached_latents"][0, 1]
    _, expected = reader_runtime.evaluate(reader, [batch], "cpu", 3)
    actual = cli.evaluate_view(reader, LatentCache(root), [(r, None) for r in batch["raw"]], "M", {}, 2, 3)
    for row, old in zip(batch["raw"], expected):
        got = actual[row_key(row)]
        assert got["nll"] == pytest.approx(old["nll"], abs=1e-6)
        assert got["pred"] == old["pred"]
        assert len(got["gold_token_nll"]) == 2


def test_chunked_bootstrap_retains_original_seeded_result():
    a, b = np.arange(71)/100, np.zeros(71)
    d = a-b
    means = d[np.random.default_rng(17).integers(0, 71, size=(137, 71))].mean(axis=1)
    got = paired_bootstrap(a, b, n_resamples=137, seed=17)
    assert got == {"delta": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
                   "hi": float(np.percentile(means, 97.5)), "n": 71}


def test_export_to_actual_training_and_reject_invalid_artifacts(tmp_path, monkeypatch):
    audit = script("build_answer_evidence_targets")
    cli = script("run_reader_experiment")
    release = tmp_path/"release"
    release.mkdir()
    (release/"adapter_model.safetensors").write_bytes(b"test fixture, not real released weights")
    def factory(cfg, cache_hidden=None):
        base = tiny_base()
        # Controlled free-generation text exercises eligibility; forward/backward
        # still use a real Mistral + LoRA. This is not a model-quality test.
        def generated(inputs_embeds, **kwargs):
            return torch.tensor([[base.tok.stoi["Nanjing"], base.tok.stoi["one"], base.tok.eos_token_id]]
                                * len(inputs_embeds))
        base.lm.generate = generated
        return SimpleNamespace(), QuROModel(cfg, base.lm, base.tok,
            TokenEmbeddingQueryEncoder(len(base.tok), 32, 128), 8, 32)
    monkeypatch.setattr(reader_runtime, "build_model", factory)
    root = tmp_path/"cache"
    row, corpus = example()
    with LatentCacheWriter(root, CacheMetadata("test", 8, 32, "float32", doc_max_length=128)) as writer:
        for doc in corpus:
            writer.add(doc, torch.randn(8, 32), 8)
    def write(name, rows):
        path = tmp_path/name
        path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        return str(path)
    train = write("train.jsonl", [row])
    dev = write("dev.jsonl", [{**row, "id": "v", "query": "Who director ?"}])
    text = write("corpus.jsonl", [{"doc_id": d, "text": t} for d, t in corpus.items()])
    common = ["--generator_path", str(release), "--cache_dir", str(root), "--device", "cpu", "--layers", "2"]
    args = common + ["--train_file", train, "--exclude_files", dev, "--corpus", text, "--batch_size", "1"]
    export = tmp_path/"export"
    audit.main(audit.parser().parse_args(args+["--mode", "export", "--out_dir", str(export)]))
    manifest = json.loads((export/"manifest.json").read_text())
    assert manifest["complete"] and manifest["rows"] == 1
    summary = json.loads((export/"summary.json").read_text())
    assert summary["target_counts"]["raw_targets_token_different_from_gold"] == 1
    assert summary["target_counts"]["matched_targets_token_different"] == 0
    assert "A-B" in summary["contrasts"]
    audit_out = tmp_path/"audit-only"
    audit.main(audit.parser().parse_args(args+["--mode", "audit", "--limit", "1", "--out_dir", str(audit_out)]))
    assert not list(audit_out.glob("*_skd_*.jsonl"))
    assert json.loads((audit_out/"manifest.json").read_text())["outputs"] == {}
    commands = ["train", *common, "--init_source", "published", "--arm", "direct-ce",
        "--train_file", str(export/"raw_skd_all.jsonl"), "--eval_file", dev,
        "--train_target", "teacher", "--target_manifest", str(export/"manifest.json"),
        "--steps", "1", "--batch_size", "1", "--grad_accum", "1", "--eval_every", "1",
        "--save_every", "1", "--max_new_tokens", "3"]
    out = tmp_path/"trained"
    parsed = cli.parser().parse_args(commands+["--out_dir", str(out)])
    cli.train(parsed)
    trained = json.loads((out/"manifest.json").read_text())
    assert trained["train_target_counts"] == {"gold": 0, "teacher": 1}
    assert trained["train_targets_differ_from_first_gold"] == 1
    cfg, cache, reader = reader_runtime.load_runtime(parsed)
    labels = reader_runtime.dataset(str(export/"raw_skd_all.jsonl"), cfg, cache, reader)
    assert labels.rows[0]["target"] == "Nanjing"  # Evaluation always gold.
    ckpt = torch.load(out/"checkpoint_last.pt", weights_only=False)
    assert any(not torch.equal(v, dict(reader.named_parameters())[k]) for k, v in ckpt["weights"].items())
    # Actual checkpoint evaluator can load the new provenance and saved args.
    cli.eval_checkpoint(cli.parser().parse_args(["eval", "--checkpoint", str(out/"checkpoint_last.pt"),
        "--eval_file", dev, "--device", "cpu", "--out_dir", str(tmp_path/"eval")]))
    provenance = reader_runtime.identity(parsed, cfg, cache, reader)
    validation_args = (str(export/"manifest.json"), str(export/"raw_skd_all.jsonl"), dev, provenance)
    validate_target_manifest(*validation_args)
    with pytest.raises(ValueError, match="matched target arms are token-identical"):
        validate_target_manifest(validation_args[0], str(export/"answer_skd_matched.jsonl"), dev, provenance)
    for key, value in (("complete", False), ("mode", "audit")):
        altered = {**manifest, key: value}
        (export/"manifest.json").write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="complete export"):
            validate_target_manifest(*validation_args)
    (export/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_target_manifest(*validation_args[:3], {**provenance, "tokenizer_sha256": "different"})
    different_dev = write("other-dev.jsonl", [{**row, "id": "other"}])
    with pytest.raises(ValueError, match="overlap during target export"):
        validate_target_manifest(validation_args[0], validation_args[1], different_dev, provenance)
    with pytest.raises(ValueError, match="hashed teacher output"):
        validate_target_manifest(validation_args[0], train, dev, provenance)
    too_long = write("long.jsonl", [{**row, "teacher_output": "one "*49}])
    with pytest.raises(ValueError, match="truncated"):
        reader_runtime.dataset(too_long, cfg, cache, reader, target_policy="teacher")
    # Token-equivalent "teacher" data must fail before optimizer updates.
    equivalent = write("equivalent.jsonl", [{**row, "teacher_output": "Nanjing"}])
    manifest["outputs"]["raw_skd_all"]["sha256"] = file_hash(equivalent)
    (export/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="token-identical"):
        cli.train(cli.parser().parse_args(commands+["--train_file", equivalent, "--out_dir", str(tmp_path/"equal")]))
    # Held-out normalized question overlap must fail before any model loading.
    overlap = write("overlap.jsonl", [{**row, "id": "different", "query": "  WHERE   BORN ?  "}])
    with pytest.raises(ValueError, match="overlaps"):
        audit.main(audit.parser().parse_args(args+["--exclude_files", overlap, "--out_dir", str(tmp_path/"leak")]))
