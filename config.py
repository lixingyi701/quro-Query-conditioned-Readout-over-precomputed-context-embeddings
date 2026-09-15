"""QuRO v0.1 configuration.

Three components, three time scales -- do not conflate them:

    offline   frozen PISCO/COCOM        document -> (m, h) latents, cached to disk
    online    trained readout           query + K*m latents -> B soft tokens
    consumer  frozen Mistral + LoRA     B soft tokens + prompt -> answer

Only the middle one is QuRO.  The offline compressor is never loaded during
online training, and the generator is frozen apart from a LoRA adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from src import paths

__version__ = "0.1.0"


# --------------------------------------------------------------------------------------
# Online readout -- the trained part
# --------------------------------------------------------------------------------------
@dataclass
class ReadoutConfig:
    # "quro"            : query-conditioned Perceiver-IO-style readout (ours)
    # "pisco_direct"    : no selection, hand every cached latent to the decoder
    # "similarity_topb" : non-parametric top-B by cosine similarity to the query
    kind: str = "quro"

    # Latents are 4096-d in Mistral space; attending at that width costs 211M
    # parameters, so the readout works in a bottleneck and bridges back out.
    d_readout: int = 1024
    cache_hidden: Optional[int] = None          # filled from the cache manifest

    # ---- output query array: the core ablation axis (design doc 7.1) ----
    # "agnostic" : learned slots only -> query-agnostic second compression (variant A)
    # "add"      : slot + Linear(pooled query)
    # "film"     : slot * (1 + scale(q)) + shift(q)
    # "concat"   : Linear([slot; pooled query])   -> explicit e_q + P
    # "xattn"    : slots cross-attend query tokens -> slots can specialise (variant C)
    output_query_mode: str = "xattn"

    max_budget: int = 8                         # B_max, and the default B
    budget_buckets: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    adaptive_budget: bool = False

    num_blocks: int = 1                         # design doc 7.2 sweeps {1, 2, 4}
    num_heads: int = 8
    head_dim: Optional[int] = None
    cross_widening: int = 1
    self_widening: int = 2
    dropout: float = 0.0

    add_document_source: bool = True            # retrieval-rank embedding
    add_slot_index: bool = True                 # position within one document
    residual_readout: bool = True               # E = s*AttnPool(Z) + zero-init Delta
    max_document_sources: int = 32
    max_latents_per_document: int = 64

    def __post_init__(self):
        valid = {"agnostic", "add", "film", "concat", "xattn"}
        if self.output_query_mode not in valid:
            raise ValueError(f"unknown output_query_mode: {self.output_query_mode}")
        if self.kind not in {"quro", "pisco_direct", "similarity_topb"}:
            raise ValueError(f"unknown readout kind: {self.kind}")
        buckets = sorted({int(x) for x in self.budget_buckets})
        if not buckets or buckets[0] < 1:
            raise ValueError("budget_buckets must be positive integers")
        buckets = [x for x in buckets if x <= self.max_budget]
        if self.max_budget not in buckets:
            buckets.append(self.max_budget)
        self.budget_buckets = sorted(buckets)


# --------------------------------------------------------------------------------------
# Online query encoder -- frozen, and the only network run online besides the readout
# --------------------------------------------------------------------------------------
@dataclass
class QueryEncoderConfig:
    """Turns the query string into the readout's Q side.

    ``hf`` takes **per-token hidden states**, not the pooled sentence vector: a
    pooled vector gives every output slot the identical conditioning signal, which
    kills the ``xattn`` mode's ability to let different slots chase different parts
    of a multi-hop question.
    """

    kind: str = "hf"                            # "hf" | "toy"
    name_or_path: str = ""                      # blank -> paths.ENCODER_PATH
    d_model: int = 128                          # toy only
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    freeze: bool = True
    layer_index: int = -1
    add_learned_pos: bool = False               # HF backbones already carry RoPE
    max_doc_len: int = 64                       # max query tokens (HFDocEncoder field name)

    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    def __post_init__(self):
        if self.kind not in {"hf", "toy"}:
            raise ValueError(f"unknown query encoder kind: {self.kind}")
        if self.kind == "hf" and not self.name_or_path:
            self.name_or_path = paths.ENCODER_PATH


# --------------------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    kind: str = "pisco"                         # "pisco" | "toy"
    name_or_path: str = ""                      # blank -> paths.PISCO_MISTRAL
    dtype: str = "bfloat16"
    device: Optional[str] = None
    # "pisco"  : keep PISCO's trained decoder_adapter and train it further (warm start)
    # "random" : re-initialise the adapter -- the control for "was it the warm start?"
    # "frozen" : train nothing in the decoder -- isolates the readout's contribution
    lora_init: str = "pisco"

    toy_n_layer: int = 4
    toy_n_head: int = 4
    toy_d_model: int = 256
    toy_max_pos: int = 2048

    def __post_init__(self):
        if self.kind not in {"pisco", "toy"}:
            raise ValueError(f"unknown generator kind: {self.kind}")
        if self.lora_init not in {"pisco", "random", "frozen"}:
            raise ValueError(f"unknown lora_init: {self.lora_init}")
        if self.kind == "pisco" and not self.name_or_path:
            self.name_or_path = paths.PISCO_MISTRAL


@dataclass
class DecoderInputConfig:
    """What the decoder sees besides the readout's soft tokens.

    D0 pisco-identical | D1 question text removed | D2 question first | D3 slots only.
    ``query_text_dropout`` trains with D1 part of the time: with no plain-text
    query the soft tokens become the only route for query information, so the
    readout is forced to condition on the query rather than leaving the
    evidence/question matching to the decoder.
    """

    input_mode: str = "D0"
    query_text_dropout: float = 0.0

    def __post_init__(self):
        if self.input_mode not in {"D0", "D1", "D2", "D3"}:
            raise ValueError(f"unknown decoder_input_mode: {self.input_mode}")
        if not 0.0 <= self.query_text_dropout <= 1.0:
            raise ValueError("query_text_dropout must be in [0,1]")


# --------------------------------------------------------------------------------------
# Data / training
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    train_file: str = ""
    eval_files: Dict[str, str] = field(default_factory=dict)
    cache_dir: Optional[str] = None
    max_query_len: int = 64
    max_answer_len: int = 48
    max_docs: Optional[int] = None              # cap on retrieved K; None = use all
    prefer_teacher_output: bool = True

    def resolved_eval_files(self) -> Dict[str, str]:
        return dict(self.eval_files) if self.eval_files else {"train": self.train_file}

    @property
    def vocab_files(self) -> List[str]:
        """The toy tokenizer must cover every split, or held-out answers become UNK."""
        return [self.train_file] + list(self.resolved_eval_files().values())


@dataclass
class TrainConfig:
    beta_qa: float = 1.0
    budget_loss_weight: float = 0.1
    steps: int = 3000
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 20
    seed: int = 42
    device: str = "auto"
    out_dir: str = os.path.join(paths.RUNS_DIR, "debug")
    resume_from: Optional[str] = None

    bf16: bool = True
    grad_accum: int = 1
    grad_ckpt: bool = False

    # One checkpoint should serve every budget: sampling B per step enforces the
    # nested (matryoshka) slot structure that slicing slots[:B] silently assumes.
    budget_dropout: bool = True

    # Trust region on the readout's residual branch while it is still random.
    residual_weight: float = 0.1
    residual_warmup_steps: int = 300

    eval_max_samples: Optional[int] = 1000
    eval_batch_size: int = 16
    gen_max_new_tokens: int = 32
    prefer_teacher_output: bool = True


@dataclass
class Config:
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    query_encoder: QueryEncoderConfig = field(default_factory=QueryEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    decoder: DecoderInputConfig = field(default_factory=DecoderInputConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def revalidate(self):
        for section in (self.readout, self.query_encoder, self.generator, self.decoder):
            section.__post_init__()
        return self

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        r, g = self.readout, self.generator
        return (f"[cfg] readout={r.kind}/{r.output_query_mode} d_r={r.d_readout} "
                f"B={r.max_budget} buckets={r.budget_buckets} blocks={r.num_blocks} "
                f"residual={r.residual_readout} | generator={g.kind}({g.lora_init}) | "
                f"decoder_input={self.decoder.input_mode} "
                f"qdrop={self.decoder.query_text_dropout}")


# --------------------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------------------
def toy_config() -> Config:
    """CPU-only contract test: no downloads, same prompt/readout code as the real stack."""
    cfg = Config()
    cfg.readout = ReadoutConfig(d_readout=128, max_budget=8, budget_buckets=[4, 8],
                                num_heads=4, num_blocks=1, cache_hidden=64)
    cfg.query_encoder = QueryEncoderConfig(kind="toy", d_model=128)
    cfg.generator = GeneratorConfig(kind="toy", toy_d_model=64, toy_max_pos=1024,
                                    lora_init="pisco")
    cfg.data = DataConfig(train_file="data/toy/train.jsonl", max_query_len=32, max_answer_len=16)
    cfg.train = TrainConfig(steps=50, batch_size=2, device="cpu", bf16=False,
                            out_dir="runs/toy", eval_max_samples=8)
    return cfg


def _pisco_base(tag: str) -> Config:
    cfg = Config()
    cfg.readout = ReadoutConfig(
        kind="quro", d_readout=1024, cache_hidden=4096, output_query_mode="xattn",
        max_budget=8, budget_buckets=[4, 8], num_blocks=1, num_heads=8,
        residual_readout=True)
    cfg.query_encoder = QueryEncoderConfig(
        kind="hf", name_or_path=paths.ENCODER_PATH, dtype="bfloat16",
        freeze=True, max_doc_len=64)
    cfg.generator = GeneratorConfig(kind="pisco", name_or_path=paths.PISCO_MISTRAL,
                                    dtype="bfloat16", lora_init="pisco")
    cfg.decoder = DecoderInputConfig(input_mode="D0", query_text_dropout=0.0)
    cfg.train = TrainConfig(out_dir=os.path.join(paths.RUNS_DIR, tag))
    return cfg


def pisco_smoke_config() -> Config:
    """Level 0: prove the wiring end to end in minutes, not hours."""
    cfg = _pisco_base("smoke")
    root = os.path.join(paths.DATA_DIR, "smoke")
    cfg.data = DataConfig(
        train_file=os.path.join(root, "train.jsonl"),
        eval_files={"dev": os.path.join(root, "dev.jsonl")},
        cache_dir=os.path.join(paths.CACHE_ROOT, "smoke-pisco-r16"),
        max_docs=5)
    cfg.train.steps = 100
    cfg.train.batch_size = 4
    cfg.train.eval_max_samples = 64
    cfg.train.residual_warmup_steps = 20
    return cfg


def pisco_gonogo_config() -> Config:
    """Level 1: the A-vs-C go/no-go gate (design doc 10, steps 1-2)."""
    cfg = _pisco_base("gonogo")
    root = os.path.join(paths.DATA_DIR, "gonogo")
    cfg.data = DataConfig(
        train_file=os.path.join(root, "train.jsonl"),
        # ``dev`` is in-domain but 43% yes/no, where a constant "Yes" already
        # scores 6.8% EM -- too weak to decide anything on its own.  TriviaQA has
        # a 0.35% constant floor and two-word factoid answers with alias lists,
        # so it is the discriminative split and the one to read first.
        eval_files={"dev": os.path.join(root, "dev.jsonl"),
                    "trivia": os.path.join(paths.DATA_DIR, "trivia/queries.jsonl")},
        cache_dir=os.path.join(paths.CACHE_ROOT, "gonogo-pisco-r16"),
        max_docs=10)
    cfg.train.steps = 3000
    cfg.train.batch_size = 8
    cfg.train.grad_accum = 2
    cfg.train.eval_max_samples = 1000
    return cfg


PRESETS = {
    "toy": toy_config,
    "pisco_smoke": pisco_smoke_config,
    "pisco_gonogo": pisco_gonogo_config,
}


def parse_eval_files(spec: str) -> Dict[str, str]:
    """Parse ``--eval_files "dev=a.jsonl,test=b.jsonl"``."""
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f'--eval_files needs name=path, got: "{chunk}"')
        name, path = (x.strip() for x in chunk.split("=", 1))
        if not name or not path:
            raise ValueError(f'--eval_files name and path must both be non-empty: "{chunk}"')
        out[name] = path
    if not out:
        raise ValueError("--eval_files parsed to nothing")
    return out


def get_config(preset: str = "toy") -> Config:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset={preset}; choose from {list(PRESETS)}")
    return PRESETS[preset]()
