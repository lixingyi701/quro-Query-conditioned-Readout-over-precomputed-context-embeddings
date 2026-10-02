"""Reset protocol: real tiny decoder, zero-init identity, causal/KV correctness."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_reader_experiment import tiny_base, sample_batch
from src.reader_experiment import ReaderExperiment, ReaderSpec
from src import reader_runtime
from src.cache import CacheMetadata, LatentCacheWriter
from src.model import QuROModel, TokenEmbeddingQueryEncoder


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reader(arm):
    base = tiny_base(dropout=.2)
    model = ReaderExperiment(base, ReaderSpec(arm, (2,), cross_dim=8, cross_heads=2))
    model.freeze_decoder()
    return model, sample_batch(base.tok)


@pytest.mark.parametrize("arm", ["direct-read", "direct-mlp"])
def test_identity_frozen_gradients_and_causality(arm):
    model, batch = reader(arm)
    model.train()
    assert not model.lm.training
    initial = {k: v.detach().clone() for k, v in model.lm.state_dict().items()}
    packed = model.pack(batch)
    baseline = model.lm(**packed["inputs"], use_cache=False).logits
    with model.activate(packed):
        output = model.lm(**packed["inputs"], use_cache=False)
        torch.testing.assert_close(output.logits, baseline, rtol=0, atol=0)
        output.loss.backward()
    assert all(p.grad is None for p in model.lm.parameters())
    assert model.cross["2"].out.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    optimizer.step(); optimizer.zero_grad()
    # After opening the gate, gradients must reach input projections too.
    packed = model.pack(batch)
    with model.activate(packed):
        model.lm(**packed["inputs"], use_cache=False).loss.backward()
    projection = model.cross["2"].v if arm == "direct-read" else model.cross["2"].up
    assert projection.weight.grad.abs().sum() > 0
    for name, value in model.lm.state_dict().items():
        torch.testing.assert_close(initial[name], value, rtol=0, atol=0)
    # No writing into memory/query; no future answer leakage into first predictor.
    with model.activate(packed):
        logits = model.lm(**packed["inputs"], use_cache=False).logits
    changed = dict(batch)
    changed["target_ids"] = [[batch["target_ids"][1][0], model.tok.eos_token_id],
                             [batch["target_ids"][0][0], model.tok.eos_token_id, model.tok.eos_token_id]]
    other = model.pack(changed)
    with model.activate(other):
        counterfactual = model.lm(**other["inputs"], use_cache=False).logits
    for i, pos in enumerate(packed["anchor"]):
        torch.testing.assert_close(logits[i, :pos], baseline[i, :pos], atol=0, rtol=0)
        torch.testing.assert_close(logits[i, pos], counterfactual[i, pos], atol=0, rtol=0)
    weights = model.checkpoint_state()
    assert weights and all(not n.startswith("lm.") for n in weights)
    restored, _ = reader(arm)
    restored.restore_state(weights)
    p = restored.pack(batch)
    with restored.activate(p):
        actual = restored.lm(**p["inputs"], use_cache=False).logits
    torch.testing.assert_close(actual, logits, atol=0, rtol=0)


@pytest.mark.parametrize("arm", ["direct-read", "direct-mlp"])
def test_nonzero_reader_cached_generation_matches_full(arm):
    model, batch = reader(arm)
    torch.nn.init.normal_(model.cross["2"].out.weight, std=.1)
    model.eval()
    p = model.pack(batch, targets=False, pad_side="left")
    mask = p["inputs"]["attention_mask"]
    with torch.no_grad(), model.activate(p):
        output = model.lm(**p["inputs"], position_ids=(mask.cumsum(1)-1).clamp_min(0), use_cache=True)
        token = output.logits[:, -1].argmax(-1)
        mask2 = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        cached = model.lm(input_ids=token[:, None], attention_mask=mask2,
                          position_ids=mask.sum(1)[:, None], past_key_values=output.past_key_values,
                          use_cache=True).logits[:, -1]
    full = dict(p)
    full["inputs"] = {"inputs_embeds": torch.cat([p["inputs"]["inputs_embeds"],
                       model.lm.get_input_embeddings()(token[:, None])], 1),
                       "attention_mask": mask2, "position_ids": (mask2.cumsum(1)-1).clamp_min(0)}
    with torch.no_grad(), model.activate(full):
        expected = model.lm(**full["inputs"], use_cache=False).logits[:, -1]
    torch.testing.assert_close(cached, expected, atol=1e-6, rtol=1e-4)
    assert len(model.generate(batch, 3)) == 2


def test_branch_intervention_keeps_original_input():
    model, batch = reader("direct-read")
    torch.nn.init.normal_(model.cross["2"].out.weight, std=.1)
    diag = script("diagnose_reader_experiment")
    ordinary = model.pack(batch)
    donor_z = torch.randn_like(batch["cached_latents"])
    fake_cache = SimpleNamespace(get_many=lambda *args, **kwargs: (donor_z, batch["document_mask"], None))
    mapping = [{"id": r["id"], "donor_docs": ["wrong"]} for r in batch["raw"]]
    with diag.branch_mismatch(model, fake_cache, mapping):
        wrong = model.pack(batch)
        for key in ordinary["inputs"]:
            assert torch.equal(ordinary["inputs"][key], wrong["inputs"][key])
        with model.activate(wrong):
            wrong_logits = model.lm(**wrong["inputs"], use_cache=False).logits
    assert "branch_z" not in model.pack(batch)
    with model.activate(ordinary):
        correct = model.lm(**ordinary["inputs"], use_cache=False).logits
    assert not torch.equal(correct, wrong_logits)
    with diag.disabled_cross(model), model.activate(ordinary):
        disabled = model.lm(**ordinary["inputs"], use_cache=False).logits
    original = model.lm(**ordinary["inputs"], use_cache=False).logits
    torch.testing.assert_close(disabled, original, atol=0, rtol=0)


def test_published_cli_never_loads_p1_and_resume_freezes_decoder(tmp_path, monkeypatch):
    cli = script("run_reader_experiment")
    release = tmp_path / "release"
    release.mkdir()
    (release / "adapter_model.safetensors").write_bytes(b"mock loading only")
    def factory(cfg, cache_hidden=None):
        base = tiny_base()
        model = QuROModel(cfg, base.lm, base.tok, TokenEmbeddingQueryEncoder(len(base.tok), 32, 128), 8, 32)
        def forbidden(*args, **kwargs):
            raise AssertionError("published route tried to load a P1 checkpoint")
        model.load = forbidden
        return SimpleNamespace(), model
    monkeypatch.setattr(reader_runtime, "build_model", factory)
    cache = tmp_path / "cache"
    with LatentCacheWriter(cache, CacheMetadata("test", 8, 32, "float32", doc_max_length=128)) as w:
        for i in range(4):
            w.add(str(i), torch.randn(8, 32), 8)
    def write(name, indices):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps({"id": str(i), "query": f"Where born {i} ?",
            "answers": ["Nanjing"], "retrieved_doc_ids": [str(i)]}) for i in indices))
        return str(path)
    train, dev = write("train.jsonl", [0,1]), write("dev.jsonl", [2,3])
    args = ["train", "--init_source", "published", "--generator_path", str(release),
            "--cache_dir", str(cache), "--train_file", train, "--eval_file", dev,
            "--device", "cpu", "--layers", "2", "--batch_size", "1", "--grad_accum", "1",
            "--max_new_tokens", "2", "--cross_dim", "8", "--cross_heads", "2",
            "--eval_every", "1", "--save_every", "1"]
    for arm in ("direct-read", "direct-mlp", "direct-ce"):
        out = tmp_path / arm
        cmd = args + ["--arm", arm, "--steps", "2", "--out_dir", str(out)]
        if arm != "direct-ce":
            cmd += ["--freeze_decoder"]
        original_save = cli.save_checkpoint
        def snapshot(path, *values):
            original_save(path, *values)
            if Path(path).name == "checkpoint_last.pt" and values[3] == 1:
                import shutil
                shutil.copyfile(path, out / "step1.pt")
        monkeypatch.setattr(cli, "save_checkpoint", snapshot)
        cli.train(cli.parser().parse_args(cmd))
        before = torch.load(out / "checkpoint_last.pt", weights_only=False)
        monkeypatch.setattr(cli, "save_checkpoint", original_save)
        cli.train(cli.parser().parse_args(cmd + ["--resume", str(out / "step1.pt")]))
        after = torch.load(out / "checkpoint_last.pt", weights_only=False)
        for key, value in before["weights"].items():
            torch.testing.assert_close(value, after["weights"][key], rtol=0, atol=0)
        cli.eval_checkpoint(cli.parser().parse_args(["eval", "--checkpoint", str(out / "checkpoint_last.pt"),
            "--eval_file", dev, "--out_dir", str(out / "eval"), "--device", "cpu"]))
    with pytest.raises(ValueError, match="forbids"):
        reader_runtime.load_runtime(cli.parser().parse_args(args + ["--arm", "direct-ce", "--out_dir", "unused",
                                    "--init_checkpoint", "must-not-exist.pt"]))
    # Exercise automatic stopping and frozen step-zero checkpoints.
    original_eval = cli.evaluate
    def tied(*a, **kw):
        summary, rows = original_eval(*a, **kw)
        summary["metrics"]["all"]["substring"] = .5
        return summary, rows
    monkeypatch.setattr(cli, "evaluate", tied)
    out = tmp_path / "early"
    cli.train(cli.parser().parse_args(args + ["--arm", "direct-ce", "--steps", "10",
              "--early_stop_patience", "2", "--out_dir", str(out)]))
    assert json.loads((out/"completion.json").read_text())["completed_step"] == 2
    frozen = tmp_path / "frozen"
    cli.train(cli.parser().parse_args(args + ["--arm", "direct-ce", "--freeze_decoder", "--steps", "0",
              "--out_dir", str(frozen)]))
    assert torch.load(frozen/"checkpoint_last.pt", weights_only=False)["weights"] == {}


def test_zero_gate_rejects_behavior_change(tmp_path):
    checker = script("check_reader_reset_zero")
    settings = {"eval_samples":1,"eval_batch_size":1,"max_new_tokens":2,"attn_implementation":"eager"}
    rows = [{"id":"x","golds":["a"],"pred":"a","nll":.5}]
    for name in ("reference", "candidate"):
        root = tmp_path/name
        root.mkdir()
        (root/"manifest.json").write_text(json.dumps({"provenance":{"source":"same"},"args":settings}))
        (root/"step_zero_predictions.json").write_text(json.dumps(rows))
    checker.check(tmp_path/"reference", [tmp_path/"candidate"])
    rows[0]["pred"] = "wrong"
    (tmp_path/"candidate"/"step_zero_predictions.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="prediction"):
        checker.check(tmp_path/"reference", [tmp_path/"candidate"])


def test_reset_split_filters_content_duplicates_and_checks_cache(tmp_path):
    prep = script("prepare_reader_reset")
    def row(i, q=None):
        return {"id": str(i), "query": q or f"Question {i}", "retrieved_doc_ids": ["d"],
                "answers": ["a"], "hop_type": "bridge"}
    def write(name, rows):
        p = tmp_path/name
        p.write_text("\n".join(json.dumps(r) for r in rows))
        return str(p)
    old = write("old", [row(0),row(1)])
    pool = write("pool", [row(i) for i in range(9)] + [row(10,"  QUESTION   0 ")])
    dev = write("dev", [row(7),row(8)])
    cache = tmp_path/"manifest.json"
    cache.write_text(json.dumps({"latent_size":8,"hidden_size":4096,"doc_max_length":128,
                                "compr_rate":16,"documents":{"d":{}}}))
    args = SimpleNamespace(seen=old, pool=pool, exclude=[dev], cache_manifest=str(cache),
                           tune_size=2, seed=42, out_dir=str(tmp_path/"out"))
    report = prep.prepare(args)
    assert report["outputs"]["old_train.jsonl"]["rows"] == 2
    assert report["outputs"]["new_train_matched.jsonl"]["rows"] == 2
    assert any(r["id"] == "10" for r in report["excluded_rows"])
    with pytest.raises(ValueError, match="overwrite"):
        prep.prepare(args)
    args.out_dir = str(tmp_path/"bad")
    cache.write_text(json.dumps({"latent_size":8,"hidden_size":4096,"doc_max_length":128,
                                "compr_rate":16,"documents":{}}))
    with pytest.raises(ValueError, match="missing"):
        prep.prepare(args)
