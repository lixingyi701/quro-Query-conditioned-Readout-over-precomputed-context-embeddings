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
    # "agnostic_matched" : as xattn, but cross-attends a fixed learned placeholder
    #              instead of the query -> query-agnostic *and* parameter-matched to C
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
    residual_readout: bool = True               # legacy alias; False == output_mode "delta_only"
    # Which terms of E = s*AttnPool(Z) + Delta reach the decoder.  Separate from
    # the initialisation below on purpose: the old residual_readout flag moved
    # both at once, so a drop under it was unattributable (warning_and_target W2).
    #   "full"       : s*AttnPool(Z) + Delta
    #   "pool_only"  : s*AttnPool(Z)          -> pure selection; carries the "readout" claim
    #   "delta_only" : Delta                  -> free branch; output is synthesised
    output_mode: str = "full"
    # "zeros" makes Delta exactly 0 at step 0; "default" is PyTorch's Linear init.
    # None picks zeros when a pooling branch exists, default otherwise.
    out_proj_init: Optional[str] = None
    # Start the *scoring* at a working rule, the way residual_readout starts the
    # *output* at one.  Query-latent cosine is added to the first block's attention
    # logits, so step 0 attends where non-parametric top-B would.  Without it a
    # random to_q/to_k pair yields tiny, semantically meaningless logits and the
    # softmax averages instead of selecting (measured: 99.85% of uniform entropy).
    cosine_prior: bool = True
    # "rank"   : slot b centres on the b-th ranked candidate -> distinct slots
    # "shared" : every slot centres on the best candidate    -> identical slots
    # Kept switchable because "do the B slots need to differ?" is an empirical
    # question, not an assumption: xRAG answers from a single soft token, so slot
    # redundancy may cost less than it looks.
    prior_mode: str = "rank"
    # The prior is row-standardised, so tau reads in standard deviations: 5 puts
    # the best candidate about five sigma above the field, which is peaked enough
    # to behave like top-B selection at step 0.
    tau_init: float = 5.0
    max_document_sources: int = 32
    max_latents_per_document: int = 64

    def __post_init__(self):
        valid = {"agnostic", "agnostic_matched", "add", "film", "concat", "xattn"}
        if self.output_query_mode not in valid:
            raise ValueError(f"unknown output_query_mode: {self.output_query_mode}")
        if self.kind not in {"quro", "pisco_direct", "similarity_topb"}:
            raise ValueError(f"unknown readout kind: {self.kind}")
        if self.prior_mode not in {"rank", "shared"}:
            raise ValueError(f"unknown prior_mode: {self.prior_mode}")
        if self.output_mode not in {"full", "pool_only", "delta_only"}:
            raise ValueError(f"unknown output_mode: {self.output_mode}")
        if self.out_proj_init not in {None, "zeros", "default"}:
            raise ValueError(f"unknown out_proj_init: {self.out_proj_init}")
        # Legacy runs and scripts pass residual_readout=False and expect Delta alone.
        if not self.residual_readout and self.output_mode == "full":
            self.output_mode = "delta_only"
        self.residual_readout = (self.output_mode == "full")
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

    kind: str = "generator"                     # "generator" | "hf" | "toy"
    name_or_path: str = ""                      # blank -> paths.ENCODER_PATH
    d_model: int = 128                          # toy only
    # Sentence vector for the cosine prior.  "mean" is the measured winner, not
    # the a-priori choice: the decoder-only convention is last-token pooling
    # (E5-Mistral, RepLLaMA), but here last scores 60.1% gold targeting against
    # 67.6% for mean.  The convention does not transfer, so the default follows
    # the measurement.
    pooling: str = "mean"                       # "mean" | "last" | "weighted"
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    # ``freeze`` only means "no gradient flows back through the query path".  It
    # does NOT mean the query representation is constant: with kind="generator"
    # the encoder and the decoder are the same object, so the decoder's LoRA
    # updates change what the query encodes into.  The three states are distinct
    # and must not share one boolean (HANDOFF.md §4 W3):
    #
    #   "shared_current"  query uses whatever the decoder adapter currently is.
    #                     Historical behaviour; the representation drifts during
    #                     training, which is a design choice, not a bug -- but it
    #                     is not what "frozen query encoder" describes.
    #   "fixed_adapter"   query uses a frozen copy of the adapter taken at init;
    #                     the decoder trains its own.  The query representation is
    #                     then genuinely a fixed function.
    #
    # Neither is assumed better.  fixed_adapter removes a confound; whether it
    # helps accuracy is an experiment.
    representation: str = "shared_current"
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
        if self.kind not in {"generator", "hf", "toy"}:
            raise ValueError(f"unknown query encoder kind: {self.kind}")
        if self.pooling not in {"last", "mean", "weighted"}:
            raise ValueError(f"unknown query pooling: {self.pooling}")
        if self.representation not in {"shared_current", "fixed_adapter"}:
            raise ValueError(f"unknown query representation: {self.representation}")
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
    # Slots per document block in the prompt.  PISCO exposes it; COCOM v1 does
    # not, and its config has no doc_max_length to derive it from.
    n_mem_tokens: Optional[int] = None

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
    # D4/D5: how many embeddings the question is compressed into.  Questions
    # average ~23 tokens on HotpotQA, so 6 is roughly a 4x ratio -- matching what
    # the documents already get, and the point of the design: if documents can be
    # compressed, so can the question.
    query_tokens: int = 6
    query_text_dropout: float = 0.0

    def __post_init__(self):
        # Derived from the prompt builder rather than repeated.  A second copy of
        # this list is how D4/D5 were accepted by the CLI and then rejected here,
        # and how the --preset list hid pisco_hotpot before it.
        from src.prompt import DECODER_INPUT_MODES

        if self.input_mode not in DECODER_INPUT_MODES:
            raise ValueError(
                f"unknown decoder_input_mode: {self.input_mode}; "
                f"expected one of {DECODER_INPUT_MODES}")
        if self.input_mode in ("D4", "D5") and self.query_tokens < 1:
            raise ValueError(f"{self.input_mode} compresses the question into "
                             "query_tokens embeddings; it must be >= 1")
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
    # The readout starts from scratch and the decoder LoRA starts from PISCO's
    # trained weights, but they have shared one learning rate.  A separate rate
    # for the decoder lets the readout move without dragging a warm-started
    # adapter at the same speed.  None = use ``lr``, i.e. today's behaviour.
    decoder_lr: Optional[float] = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    log_every: int = 20
    # Validate during training instead of only scoring the final step.  Without
    # this, a run that peaked at step 1500 and then overfitted is indistinguishable
    # from one that never got there, and "3000 steps was not enough" cannot be
    # told apart from "the capacity is not there".  0 disables.
    eval_every: int = 0
    # How many dev rows the interval validation uses.  Small on purpose: it runs
    # many times and only has to rank checkpoints, not produce a reportable number.
    eval_every_samples: int = 500
    # Which metric picks the best checkpoint.  Fixed before the run, so it cannot
    # be chosen after seeing the curves.
    select_metric: str = "em"
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

    # DataLoader prefetch workers.  At m=32 one batch pulls 21 MB out of the
    # memmap and with 0 workers that read blocks the training step: measured
    # 2.88 s/step at 0 workers against 0.67 s/step at 4, with the GPUs idling in
    # between.  Default stays 0 so small runs keep a single process.
    num_workers: int = 0
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
                f"out={r.output_mode}/{r.out_proj_init or 'auto'} | "
                f"generator={g.kind}({g.lora_init}) | "
                f"decoder_input={self.decoder.input_mode} "
                f"qdrop={self.decoder.query_text_dropout}")


# --------------------------------------------------------------------------------------
# Arm taxonomy
# --------------------------------------------------------------------------------------
# Query information reaches the readout by two independent routes, and the
# historical runs moved only one of them.  Every A arm on disk has
# ``cosine_prior=True`` (audited 2026-09-16 over /data02/quro/runs/*/config.json),
# so it is A1 -- "cosine-conditioned" -- and never the query-agnostic control the
# results were reported against.  Naming the four cells makes that unstatable.
#
#   arm  cosine prior   query into output slots   what it isolates
#   A0   off            off                       genuinely query-agnostic readout
#   A1   on             off                       cosine conditioning alone
#   C0   off            on                        learned conditioning alone
#   C1   on             on                        the full method
#   S    top-B by cosine, no learnable slots
#   P    every cached latent, no second compression
ARMS = {
    "A0": {"kind": "quro", "output_query_mode": "agnostic", "cosine_prior": False},
    "A1": {"kind": "quro", "output_query_mode": "agnostic", "cosine_prior": True},
    "C0": {"kind": "quro", "output_query_mode": "xattn", "cosine_prior": False},
    "C1": {"kind": "quro", "output_query_mode": "xattn", "cosine_prior": True},
    "S": {"kind": "similarity_topb"},
    "P": {"kind": "pisco_direct"},
}


def apply_arm(cfg: Config, arm: str, param_matched: bool = False) -> Config:
    """Set every field that defines an arm, so none can be left half-specified."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    spec = dict(ARMS[arm])
    if param_matched and spec.get("output_query_mode") == "agnostic":
        spec["output_query_mode"] = "agnostic_matched"
    for key, value in spec.items():
        setattr(cfg.readout, key, value)
    cfg.revalidate()
    return cfg


def arm_label(cfg: Config) -> str:
    """Name the arm an arbitrary config actually implements, not the one it is tagged.

    Used to stamp every run record, so a mislabelled launch shows up in the results
    file rather than only in the launch script.
    """
    r = cfg.readout
    if r.kind == "similarity_topb":
        return "S"
    if r.kind == "pisco_direct":
        return "P"
    agnostic = r.output_query_mode in ("agnostic", "agnostic_matched")
    label = ("A" if agnostic else "C") + ("1" if r.cosine_prior else "0")
    if r.output_query_mode == "agnostic_matched":
        label += "m"
    return label


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
    # The query is encoded by the generator itself: PISCO latents are generator
    # hidden states, so this is the only way Q and K share a space without a
    # bridge learned from scratch.  A separate encoder measured 39.3% gold
    # targeting against 67.6% for cosine in the shared space.
    cfg.query_encoder = QueryEncoderConfig(
        kind="generator", pooling="mean", freeze=True, max_doc_len=64)
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


def pisco_hotpot_config() -> Config:
    """Multi-hop: two gold paragraphs per question, so the budget has work to do.

    TriviaQA's questions are single-fact, which is the condition least favourable
    to a multi-slot readout -- one latent answers the question and the other B-1
    slots have nothing to allocate.  HotpotQA's distractor setting asks for two
    paragraphs combined out of ten, so "which evidence, for this question" is a
    real decision rather than a one-slot one.  If query-conditioned readout does
    not help here, the B-slot design has no task left to defend it.

    ``dev`` and ``test`` are disjoint halves of the official validation split;
    read ``dev`` while iterating and leave ``test`` alone (warning_and_target §5.9).
    """
    cfg = _pisco_base("hotpot")
    root = os.path.join(paths.DATA_DIR, "hotpot")
    cfg.data = DataConfig(
        train_file=os.path.join(root, "train.jsonl"),
        eval_files={"dev": os.path.join(root, "dev.jsonl"),
                    "test": os.path.join(root, "test.jsonl")},
        cache_dir=os.path.join(paths.CACHE_ROOT, "hotpot-pisco-r16"),
        # The distractor setting supplies exactly ten paragraphs per question.
        max_docs=10)
    cfg.train.steps = 3000
    cfg.train.batch_size = 8
    cfg.train.grad_accum = 2
    cfg.train.eval_max_samples = 2000
    return cfg


PRESETS = {
    "toy": toy_config,
    "pisco_smoke": pisco_smoke_config,
    "pisco_gonogo": pisco_gonogo_config,
    "pisco_hotpot": pisco_hotpot_config,
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
