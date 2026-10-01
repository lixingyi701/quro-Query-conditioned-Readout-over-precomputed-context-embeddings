"""No model downloads: real tiny Mistral + PEFT exercise production code paths.

Run: python -m pytest -q tests/test_reader_experiment.py
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import MistralConfig, MistralForCausalLM
from peft import LoraConfig, get_peft_model
from config import get_config, apply_arm
from src.cache import CacheMetadata, LatentCache, LatentCacheWriter
from src.data import QuRODataset, QuROCollator
from src.model import QuROModel, TokenEmbeddingQueryEncoder
from src.prompt import PiscoPromptBuilder, assemble_inputs
from src.reader_experiment import (ReaderSpec, ReaderExperiment, CrossRead, StateTargets,
                                   cosine_state_loss, row_key)
from src.reader_runtime import EpochBatches
from src.toy import ToyTokenizer

torch.set_num_threads(1)


def tiny_base(seed=13, dropout=0.0):
    torch.manual_seed(seed)
    tok = ToyTokenizer(["Question", "Background", "Nanjing", "Chengdu", "Where", "Who",
                        "born", "director", "one", "two", "three", "short", "long", "?", ":"])
    config = MistralConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=512, attention_dropout=0.,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                           bos_token_id=tok.bos_token_id)
    config._attn_implementation = "eager"
    lm = get_peft_model(MistralForCausalLM(config),
                        LoraConfig(r=2, lora_alpha=4, lora_dropout=dropout,
                                   target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
                        adapter_name="decoder_adapter")
    builders = {mode: PiscoPromptBuilder(tok, 8, mode) for mode in ("D0", "AG", "RG")}
    return SimpleNamespace(lm=lm, tok=tok, pad_id=tok.pad_token_id, prompt_builders=builders)


def sample_batch(tok):
    rows = [{"id": "a", "query": "Where born ?", "retrieved_doc_ids": ["d1", "d2"],
             "answers": ["Nanjing"], "hop_type": "bridge"},
            {"id": "b", "query": "Who director one two three ?", "retrieved_doc_ids": ["d2"],
             "answers": ["Chengdu"], "hop_type": "comparison"}]
    return {"raw": rows, "ids": [r["id"] for r in rows],
            "queries": [r["query"] for r in rows],
            "retrieved_doc_ids": [r["retrieved_doc_ids"] for r in rows],
            "document_texts": [["director born Nanjing", "one two three"], ["born Chengdu"]],
            "target_ids": [[tok.stoi["Nanjing"], tok.eos_token_id],
                           [tok.stoi["Chengdu"], tok.stoi["two"], tok.eos_token_id]],
            "cached_latents": torch.randn(2, 2, 8, 32),
            "document_mask": torch.tensor([[True, True], [True, False]])}


def make_reader(arm="direct-ce", weight=.1):
    base = tiny_base()
    return base, ReaderExperiment(base, ReaderSpec(arm, (1, 2, 3), weight, 3, 16, 4)), sample_batch(base.tok)


def test_direct_identity_and_zero_state_gradients():
    base, reader, batch = make_reader()
    reader.eval()
    p = reader.pack(batch)
    flat = batch["cached_latents"].flatten(1, 2)
    mask = batch["document_mask"][:, :, None].expand(-1, -1, 8).flatten(1)
    prompts = [base.prompt_builders["D0"].build(q, int(m.sum())) for q, m in zip(batch["queries"], mask)]
    legacy = assemble_inputs(base.lm.get_input_embeddings(), prompts, flat, mask, batch["target_ids"])
    for key in legacy:
        assert torch.equal(legacy[key], p["inputs"][key])
    expected = base.lm(**legacy, use_cache=False).loss
    with reader.activate(p):
        actual = reader.lm(**p["inputs"], use_cache=False).loss
    assert torch.equal(expected, actual)
    actual.backward()
    gradients = {n: v.grad.clone() for n, v in reader.named_parameters() if v.grad is not None}
    reader.zero_grad()
    reader.spec.arm = "direct-state"
    reader.spec.state_weight = 0.
    with reader.activate(reader.pack(batch), capture=False):
        reader.lm(**p["inputs"], use_cache=False).loss.backward()
    for n, v in reader.named_parameters():
        if n in gradients:
            torch.testing.assert_close(v.grad, gradients[n], rtol=0, atol=0)


@pytest.mark.parametrize("padding", ["left", "right"])
def test_anchor_padding_and_no_answer_leak(padding):
    _, reader, batch = make_reader("direct-state")
    reader.eval()
    pre = reader.pack(batch, targets=False, pad_side=padding, raw=True)
    full = reader.pack(batch, targets=True, pad_side=padding, raw=True)
    # Force logical token positions for left-padded teacher-forced comparisons.
    for p in (pre, full):
        p["inputs"]["position_ids"] = (p["inputs"]["attention_mask"].cumsum(1) - 1).clamp_min(0)
    with torch.no_grad(), reader.activate(pre, capture=True) as c:
        reader.lm(**pre["inputs"], use_cache=False)
        a = reader.stack_states(c).clone()
    with torch.no_grad(), reader.activate(full, capture=True) as c:
        reader.lm(**full["inputs"], use_cache=False)
        b = reader.stack_states(c).clone()
    torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    for i, anchor in enumerate(full["anchor"]):
        assert full["inputs"]["labels"][i, anchor] == -100
        assert full["inputs"]["labels"][i, anchor + 1] != -100


def test_state_loss_gradient_and_teacher_detached():
    _, reader, batch = make_reader("direct-state")
    reader.eval()
    teacher = torch.randn(2, 3, 32, requires_grad=True)
    p = reader.pack(batch)
    with reader.activate(p, capture=True) as c:
        reader.lm(**p["inputs"], use_cache=False)
        loss = cosine_state_loss(reader.stack_states(c), teacher)
        loss.backward()
    assert teacher.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.lm.parameters())
    assert reader.lm.get_input_embeddings().weight.grad is None


def test_workspace_gradients_mask_and_checkpoint_roundtrip():
    _, reader, batch = make_reader("w-ce")
    reader.train()
    p = reader.pack(batch)
    assert p["wpos"].shape == (2, 3)
    assert all(int(pos[-1]) < int(s) for pos, s in zip(p["wpos"], p["anchor"]))
    with reader.activate(p):
        reader.lm(**p["inputs"], use_cache=False).loss.backward()
    assert reader.workspace.grad.abs().sum() > 0
    assert all(c.out.weight.grad.abs().sum() > 0 for c in reader.cross.values())
    assert batch["cached_latents"].grad is None
    # With zero output init, q/k/v start receiving gradients after out is updated.
    optimizer = torch.optim.SGD([p for p in reader.parameters() if p.requires_grad], lr=.1)
    optimizer.step()
    optimizer.zero_grad()
    p = reader.pack(batch)
    with reader.activate(p):
        loss = reader.lm(**p["inputs"], use_cache=False).loss
        loss.backward()
    assert all(c.v.weight.grad.abs().sum() > 0 for c in reader.cross.values())
    for c in reader.cross.values():
        w = torch.randn(2, 3, 32)
        z = torch.randn(2, 8, 32)
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[:, 4:] = False
        first = c(w, z, mask)
        z[:, 4:] = 10000
        torch.testing.assert_close(first, c(w, z, mask), rtol=0, atol=0)
    weights = reader.checkpoint_state()
    _, fresh, _ = make_reader("w-ce")
    fresh.restore_state(weights)
    reader.eval(); fresh.eval()
    for model in (reader, fresh):
        p = model.pack(batch)
        with model.activate(p):
            result = model.lm(**p["inputs"], use_cache=False).logits
        if model is reader:
            reference = result
        else:
            torch.testing.assert_close(reference, result, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["direct-state", "w-ce"])
def test_checkpointing_matches_plain_gradients(arm):
    _, reader, batch = make_reader(arm)
    # Nonzero branch tests that CA actually recomputes, not just its zero initializer.
    for cross in reader.cross.values():
        torch.nn.init.normal_(cross.out.weight, std=.02)
    saved = reader.checkpoint_state()
    target = torch.randn(2, 3, 32)
    def backward(model):
        model.train()
        packed = model.pack(batch)
        with model.activate(packed, capture=arm == "direct-state") as c:
            loss = model.lm(**packed["inputs"], use_cache=False).loss
            if arm == "direct-state":
                loss = loss + .1 * cosine_state_loss(model.stack_states(c), target)
            loss.backward()
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    plain = backward(reader)
    _, other, _ = make_reader(arm)
    other.restore_state(saved)
    other.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointed = backward(other)
    assert plain.keys() == checkpointed.keys()
    for key in plain:
        torch.testing.assert_close(plain[key], checkpointed[key], atol=1e-6, rtol=1e-4)


def test_workspace_cached_decode_equals_full_recompute():
    _, reader, batch = make_reader("w-ce")
    reader.eval()
    for cross in reader.cross.values():
        torch.nn.init.normal_(cross.out.weight, std=.2)
    p = reader.pack(batch, targets=False, pad_side="left")
    mask = p["inputs"]["attention_mask"]
    pos = (mask.cumsum(1) - 1).clamp_min(0)
    with torch.no_grad(), reader.activate(p):
        first = reader.lm(**p["inputs"], position_ids=pos, use_cache=True)
        token = first.logits[:, -1].argmax(-1)
        new_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        cached = reader.lm(input_ids=token[:, None], attention_mask=new_mask,
                           position_ids=mask.sum(1)[:, None],
                           past_key_values=first.past_key_values, use_cache=True).logits[:, -1]
    extended = dict(p)
    extended["inputs"] = {"inputs_embeds": torch.cat([p["inputs"]["inputs_embeds"],
                              reader.lm.get_input_embeddings()(token[:, None])], 1),
                          "attention_mask": new_mask,
                          "position_ids": (new_mask.cumsum(1) - 1).clamp_min(0)}
    with torch.no_grad(), reader.activate(extended):
        full = reader.lm(**extended["inputs"], use_cache=False).logits[:, -1]
    torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-4)
    assert len(reader.generate(batch, max_new_tokens=3)) == 2
    assert not any(block._forward_hooks for block in reader.blocks)


def test_cache_rejects_wrong_metadata_rows_and_incomplete(tmp_path):
    meta = {"complete": True, "split": "train", "shape": [1, 3, 32], "layers": [1, 2, 3]}
    row = {"id": "1", "query": "Q", "retrieved_doc_ids": ["a", "b"]}
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    (tmp_path / "index.json").write_text(json.dumps({row_key(row): 0}))
    np.zeros((1, 3, 32), dtype=np.float16).tofile(tmp_path / "states.bin")
    cache = StateTargets(tmp_path, {"layers": [1, 2, 3]})
    assert cache.gather([row], "cpu").shape == (1, 3, 32)
    with pytest.raises(ValueError):
        cache.gather([{**row, "retrieved_doc_ids": ["b", "a"]}], "cpu")
    with pytest.raises(ValueError):
        StateTargets(tmp_path, {"layers": [2, 3, 4]})
    meta["complete"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        StateTargets(tmp_path)


def test_sampler_resumes_same_data_across_epochs():
    full = list(EpochBatches(7, 3, 42, count=10))
    assert full[4:] == list(EpochBatches(7, 3, 42, start=4, count=6))
    assert sorted(sum(full[:3], [])) == list(range(7))


def test_comparison_pairs_by_id_and_rejects_different_conditions(tmp_path):
    spec = importlib.util.spec_from_file_location("reader_compare",
        Path(__file__).parents[1] / "scripts/compare_reader_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref, candidate = tmp_path / "ref", tmp_path / "candidate"
    ref.mkdir(); candidate.mkdir()
    meta = {"eval_file_sha256": "same", "gold_only": False, "conditions": {"decoding": "greedy"}}
    rows = [{"id": str(i), "golds": ["Nanjing"], "hop_type": "bridge",
             "substring": float(i), "em": float(i), "f1": float(i), "nll": 1.} for i in range(2)]
    for root in (ref, candidate):
        (root / "summary.json").write_text(json.dumps(meta))
    (ref / "predictions.json").write_text(json.dumps(rows))
    improved = [{**r, "substring": 1., "em": 1., "f1": 1., "nll": .5} for r in rows[::-1]]
    (candidate / "predictions.json").write_text(json.dumps(improved))
    result = module.compare(ref, candidate)["contrasts"]["all"]
    assert result["substring"]["delta"] == .5 and result["nll"]["delta"] == -.5
    (candidate / "summary.json").write_text(json.dumps({**meta, "gold_only": True}))
    with pytest.raises(ValueError):
        module.compare(ref, candidate)


def test_cli_cache_train_eval_and_resume(tmp_path, monkeypatch):
    """Execute real cache/train/eval entry points, substituting only model loading."""
    from src import reader_runtime
    spec = importlib.util.spec_from_file_location("reader_cli", Path(__file__).parents[1] / "scripts/run_reader_experiment.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = get_config("pisco_hotpot")
    apply_arm(cfg, "P")
    cfg.decoder.input_mode = "D0"
    cfg.generator.name_or_path = str(tmp_path / "generator")
    cfg.query_encoder.kind = "toy"
    cfg.query_encoder.d_model = 32
    cfg.data.cache_dir = str(tmp_path / "latent")
    cfg.data.train_file = str(tmp_path / "train.jsonl")
    cfg.data.prefer_teacher_output = True
    cfg.readout.cache_hidden = 32
    def model_factory(config, cache_hidden=None):
        base = tiny_base(dropout=.1)
        model = QuROModel(config, base.lm, base.tok,
                          TokenEmbeddingQueryEncoder(len(base.tok), 32, 128), 8, 32)
        return SimpleNamespace(lm=base.lm), model
    monkeypatch.setattr(reader_runtime, "build_model", model_factory)
    _, p1 = model_factory(cfg)
    p1_path = tmp_path / "p1.pt"
    p1.save(p1_path, step=3000)
    with LatentCacheWriter(cfg.data.cache_dir,
            CacheMetadata("synthetic", 8, 32, "float32", doc_max_length=128)) as writer:
        for doc in (f"d{i}" for i in range(10)):
            writer.add(doc, torch.randn(8, 32), source_token_count=10)
    rows = [{"id": f"q{i}", "query": "Where born ?", "retrieved_doc_ids": [f"d{2*i}", f"d{2*i+1}"],
             "answers": ["Nanjing"], "teacher_output": "wrong", "gold_ranks": [0, 1],
             "hop_type": "bridge"} for i in range(5)]
    Path(cfg.data.train_file).write_text("\n".join(json.dumps(r) for r in rows[:3]))
    ev = tmp_path / "dev.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in rows[3:]))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"doc_id": f"d{i}", "text": "born Nanjing"}) for i in range(10)))
    common = ["--init_checkpoint", str(p1_path), "--device", "cpu", "--layers", "1,2,3", "--batch_size", "2"]
    teacher = tmp_path / "teacher"
    ca = cli.parser().parse_args(["cache", *common, "--out_dir", str(teacher), "--corpus", str(corpus)])
    cli.build_targets(ca)
    for arm in ("direct-ce", "direct-state", "w-ce"):
        out = tmp_path / arm
        argv = ["train", *common, "--arm", arm, "--out_dir", str(out),
                "--eval_file", str(ev), "--steps", "2", "--grad_accum", "1",
                "--eval_every", "1", "--save_every", "1", "--max_new_tokens", "2",
                "--workspace_tokens", "3", "--cross_dim", "16", "--cross_heads", "4",
                "--grad_checkpointing"]
        if arm == "direct-state":
            argv += ["--teacher_cache", str(teacher)]
        a = cli.parser().parse_args(argv)
        # Keep the step-one checkpoint to test exact optimizer+RNG+sampler continuation.
        original_save = cli.save_checkpoint
        def snapshot(path, *args, **kwargs):
            original_save(path, *args, **kwargs)
            if Path(path).name == "checkpoint_last.pt" and args[3] == 1:
                import shutil
                shutil.copyfile(path, out / "step1.pt")
        monkeypatch.setattr(cli, "save_checkpoint", snapshot)
        cli.train(a)
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["train_target_counts"] == {"gold": 3, "teacher": 0}
        last = torch.load(out / "checkpoint_last.pt", weights_only=False)
        ea = cli.parser().parse_args(["eval", "--checkpoint", str(out / "checkpoint_best.pt"),
            "--eval_file", str(ev), "--out_dir", str(out / "evaluation"), "--device", "cpu", "--gold_only"])
        cli.eval_checkpoint(ea)
        assert (out / "evaluation" / "predictions.json").exists()
        monkeypatch.setattr(cli, "save_checkpoint", original_save)
        if arm == "direct-ce":
            legacy = torch.load(out / "step1.pt", weights_only=False)
            legacy["args"].pop("train_target")
            torch.save(legacy, out / "step1.pt")
        ra = cli.parser().parse_args(argv + ["--resume", str(out / "step1.pt")])
        cli.train(ra)
        resumed = torch.load(out / "checkpoint_last.pt", weights_only=False)
        for name in last["weights"]:
            torch.testing.assert_close(last["weights"][name], resumed["weights"][name], rtol=0, atol=0)
        diag_spec = importlib.util.spec_from_file_location("reader_diag",
            Path(__file__).parents[1] / "scripts/diagnose_reader_experiment.py")
        diag = importlib.util.module_from_spec(diag_spec)
        diag_spec.loader.exec_module(diag)
        if arm == "w-ce":
            da = diag.parser().parse_args(["evidence", "--checkpoint", str(out / "checkpoint_last.pt"),
                "--eval_file", str(ev), "--out_dir", str(out / "diag"), "--device", "cpu"])
            diag.main(da)
            report = json.loads((out / "diag/summary.json").read_text())
            assert set(report["conditions"]) == {"correct", "mismatch", "disabled_cross"}
        if arm == "direct-state":
            da = diag.parser().parse_args(["state", "--checkpoints", str(out / "step1.pt"),
                str(out / "checkpoint_last.pt"), "--teacher_cache", str(teacher),
                "--out_dir", str(out / "diag"), "--limit", "2", "--grad_batches", "1", "--device", "cpu"])
            diag.main(da)
            report = json.loads((out / "diag/summary.json").read_text())
            assert len(report["reports"]) == 2
            assert report["reports"][0]["gradients"][0]["state_grad_norm"] > 0
    # P1 target preference affects training only; eval NLL still uses gold.
    p1run = tmp_path / "p1target"
    args = cli.parser().parse_args(["train", *common, "--arm", "direct-ce", "--train_target", "p1",
        "--out_dir", str(p1run), "--eval_file", str(ev), "--steps", "0", "--max_new_tokens", "2"])
    cli.train(args)
    assert json.loads((p1run / "manifest.json").read_text())["train_target_counts"] == {"gold": 0, "teacher": 3}
    cfg2, cache2, reader2 = reader_runtime.load_runtime(args)
    train_ds = reader_runtime.dataset(cfg.data.train_file, cfg2, cache2, reader2, target_policy="p1")
    eval_ds = reader_runtime.dataset(str(ev), cfg2, cache2, reader2)
    assert train_ds.rows[0]["target"] == "wrong"
    assert eval_ds.rows[0]["target"] == "Nanjing"


def test_diagnostic_donors_and_gradient_statistics():
    spec = importlib.util.spec_from_file_location("reader_diag",
        Path(__file__).parents[1] / "scripts/diagnose_reader_experiment.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    rows = [{"id": str(i), "query": "q", "answers": ["a"], "retrieved_doc_ids": [str(i)]}
            for i in range(4)]
    changed, mapping = diag.mismatch_rows(rows)
    assert len({m["donor_id"] for m in mapping}) == 4
    for old, new in zip(rows, changed):
        assert old["id"] == new["id"] and old["answers"] == new["answers"]
        assert not set(old["retrieved_doc_ids"]) & set(new["retrieved_doc_ids"])
    with pytest.raises(ValueError):
        diag.mismatch_rows([rows[0], {**rows[1], "retrieved_doc_ids": ["0"]}])
    x = torch.tensor([1., 2.], requires_grad=True)
    g = diag.gradient_stats(x.sum(), -2 * x.sum(), [x], .1)
    assert g["gradient_cosine"] == pytest.approx(-1.)
    assert g["weighted_state_to_ce_ratio"] == pytest.approx(.2)
    _, reader, _ = make_reader("w-ce")
    saved = [block.gate.clone() for block in reader.cross.values()]
    with pytest.raises(RuntimeError):
        with diag.disabled_cross(reader):
            assert all(block.gate == 0 for block in reader.cross.values())
            raise RuntimeError("check finally")
    assert all(torch.equal(before, block.gate) for before, block in zip(saved, reader.cross.values()))
