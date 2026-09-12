"""QuRO v0.0 configuration.

Production runs read frozen PISCO/COCOM document latents from disk and train
the query-conditioned readout plus generator LoRA.  ``tiny`` retains a local
Perceiver document encoder solely for executable contract tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json
import os

__version__ = "0.0.0"


# ######################################################################################
# ★★★ 改这里：服务器上的模型与数据路径 ★★★
# 也可以不改文件，直接用环境变量覆盖：ENCODER_PATH=... GENERATOR_PATH=... bash scripts/run_qwen3emb.sh
# ######################################################################################

# 在线 query 编码器。缓存模式下它不参与文档压缩。
ENCODER_PATH = os.environ.get("ENCODER_PATH", "/path/to/Qwen3-Embedding-0.6B")

# 生成器：读压缩软 token 并说出答案的 causal LM。必须是能生成的模型，不能填 Embedding 模型。
# 效果不够时的升级阶梯：Qwen2.5-1.5B-Instruct -> Qwen2.5-3B-Instruct -> Qwen2.5-7B-Instruct
GENERATOR_PATH = os.environ.get("GENERATOR_PATH", "/path/to/Qwen2.5-1.5B-Instruct")

# 真实 RAG 数据（jsonl，每行 {"question": str, "answer": [str, ...], "documents": [str, ...]}）
TRAIN_FILE = os.environ.get("RAG_TRAIN_FILE", "data/rag/train.jsonl")
EVAL_FILES = {
    "dev": os.environ.get("RAG_DEV_FILE", "data/rag/dev.jsonl"),
    "test": os.environ.get("RAG_TEST_FILE", "data/rag/test.jsonl"),
}

# ######################################################################################


# --------------------------------------------------------------------------------------
# 模型结构
# --------------------------------------------------------------------------------------
@dataclass
class PerceiverConfig:
    # ---- latent bottleneck（全量压缩的容量）----
    num_latents: int = 128            # m：缓存中每篇文档的 latent 数；prototype 模式也使用它
    d_latent: int = 256               # latent 维度

    # ---- Encode：cross-attn(Q=latents, K/V=doc tokens) ----
    enc_num_heads: int = 8
    enc_head_dim: Optional[int] = None        # None -> d_latent // num_heads
    enc_num_blocks: int = 1                   # 重复 [cross + self*k] 的次数（Perceiver IO 的 repeat）
    enc_self_per_block: int = 2               # 每个 block 内 cross 之后跟几层 latent self-attn
    enc_share_weights: bool = True            # 重复 block 时是否共享权重（Perceiver IO 默认共享）

    # ---- Process：latent self-attn ----
    proc_num_layers: int = 4
    proc_num_heads: int = 8

    # ---- Decode：cross-attn(Q=output query, K/V=latents) ----
    dec_num_heads: int = 8
    dec_head_dim: Optional[int] = None
    # v0.0 中该值是最大在线预算 B_max；固定预算时也作为默认 B。
    num_compressed: int = 8
    budget_buckets: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    adaptive_budget: bool = False
    add_document_source: bool = True
    max_document_sources: int = 32
    # 外部缓存 latent 的 h。None 表示与 d_latent 相同；不同时用线性层适配。
    cached_hidden_size: Optional[int] = None

    # ---- Output query 的构造方式（最关键的消融维度）----
    # "agnostic"   : 仅 P 个可学习 slot，完全不看 query   -> 隐式/无 query 基线
    # "add"        : slot + Linear(mean-pool(query))      -> Option A，最轻
    # "film"       : slot * scale(q) + shift(q)           -> 调制式
    # "concat"     : Linear([slot; pooled query])          -> explicit query-slot concatenation
    # "xattn"      : slot 先 cross-attend query tokens     -> semantic query slots
    output_query_mode: str = "xattn"

    # ---- 通用 ----
    dropout: float = 0.0
    ff_widening: int = 4                      # FeedForward 扩张倍数（Perceiver IO: cross=1, self=4）
    cross_ff_widening: int = 1

    def __post_init__(self):
        assert self.output_query_mode in {"agnostic", "add", "film", "concat", "xattn"}, \
            f"未知的 output_query_mode: {self.output_query_mode}"
        buckets = sorted(set(int(x) for x in self.budget_buckets))
        if not buckets or buckets[0] < 1:
            raise ValueError("budget_buckets 必须是正整数列表")
        if buckets[-1] > self.num_compressed:
            # num_compressed 是 B_max；自动扩展可避免默认 buckets 与小型预设冲突。
            buckets = [x for x in buckets if x <= self.num_compressed]
            if self.num_compressed not in buckets:
                buckets.append(self.num_compressed)
        self.budget_buckets = buckets


@dataclass
class DocEncoderConfig:
    """
    Query token encoder, and prototype-only document token encoder.

    "standalone"    : 自己的 embedding + 可学习位置编码（轻、快，tiny 用它跑通链路）
    "generator_emb" : 复用冻结 generator 的 input embedding（ICAE/COCOM 路线，表示空间同源）
    "hf_encoder"    : 独立的 HF 编码器（Qwen3-Embedding-0.6B），取 last_hidden_state。
                      **重点**：取的是逐 token 的隐状态 (B, L, d)，不是池化后的那一个句向量——
                      池化向量等价于 P=1 且 query 无关，那正好是我们的对照组而不是方法本身。
    """

    kind: str = "standalone"
    d_model: int = 256                # 仅 standalone 生效；其余情况自动等于上游模型的 hidden size
    max_doc_len: int = 1024           # 文档最长 token 数（Encode 的 K/V 长度上限）
    learned_pos: bool = True

    # ---- kind="hf_encoder" 专用 ----
    name_or_path: str = ""            # 留空则回落到 ENCODER_PATH
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    freeze: bool = True               # 默认冻结：只训压缩器，才谈得上"压缩器高效"
    layer_index: int = -1             # 取第几层隐状态，-1 = 最后一层
    add_learned_pos: bool = False     # HF 模型自带位置编码，默认不再叠一层

    # ---- LoRA（默认关，--lora_doc_encoder 打开）----
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    def __post_init__(self):
        assert self.kind in {"standalone", "generator_emb", "hf_encoder"}, \
            f"未知的 doc encoder: {self.kind}"
        if self.kind == "hf_encoder" and not self.name_or_path:
            self.name_or_path = ENCODER_PATH
        if self.lora and self.kind != "hf_encoder":
            raise ValueError("LoRA 只对 kind=hf_encoder 有意义")
        if self.lora and self.freeze:
            # 冻结底座 + 只训 LoRA 是正常组合，这里只是把语义说清楚：freeze 指底座权重
            pass


@dataclass
class GeneratorConfig:
    # "toy" -> 内置随机初始化的小 causal LM；否则填 HF model id / 本地路径
    name_or_path: str = "toy"
    dtype: str = "float32"            # 真模型上建议 "bfloat16"
    freeze: bool = True               # 冻结 generator（Stage1 默认冻结）
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None   # 如 "flash_attention_2"
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    # toy 专用
    toy_n_layer: int = 4
    toy_n_head: int = 4
    toy_d_model: int = 256
    toy_max_pos: int = 2048


@dataclass
class ProjectorConfig:
    kind: str = "mlp"                 # "identity" | "linear" | "mlp"
    hidden_mult: int = 2
    dropout: float = 0.0


# --------------------------------------------------------------------------------------
# 数据 / 训练
# --------------------------------------------------------------------------------------
@dataclass
class DataConfig:
    train_file: str = "data/multiq/train.jsonl"
    # 多个评测集，name -> path。约定：
    #   seen   = 训练见过的文档 + 留出的 query   -> 测 query 路由（文档记忆无用）
    #   unseen = 完全没见过的文档               -> 测泛化
    # 留空 -> 退化成在训练集上自测（过拟合自检）
    eval_files: Dict[str, str] = field(default_factory=dict)
    max_doc_len: int = 1024
    max_query_len: int = 64
    max_answer_len: int = 32
    qa_prompt_template: str = "\nQuestion: {query}\nAnswer:"

    # ---- 数据格式 ----
    # "auto"      : 看到 question/documents 字段就按真实 RAG 格式解析，否则按合成格式
    # "rag"       : {"question": str, "answer": [str,...], "documents": [str,...]}
    # "synthetic" : {"document": str, "query": str, "answer": str, "gold_char_span": [s,e]}
    fmt: str = "auto"
    max_docs: int = 1                  # 取检索结果的前 k 篇拼接；>1 时其余篇自动成为归因的干扰项
    doc_join: str = "\n\n"
    # cache-first 格式使用 retrieved_doc_ids；原文 documents 仅用于建缓存或 prototype 模式。
    cache_dir: Optional[str] = None
    prefer_teacher_output: bool = True

    def __post_init__(self):
        assert self.fmt in {"auto", "rag", "synthetic"}, f"未知的数据格式: {self.fmt}"

    def resolved_eval_files(self) -> Dict[str, str]:
        return dict(self.eval_files) if self.eval_files else {"train": self.train_file}

    @property
    def vocab_files(self) -> List[str]:
        """toy tokenizer 建词表要覆盖所有 split，否则留出文档里的答案会变成 UNK。"""
        return [self.train_file] + list(self.resolved_eval_files().values())


@dataclass
class TrainConfig:
    stage: str = "stage1"              # v0.0 only: cache-first sequence training
    beta_qa: float = 1.0               # QA next-token loss 权重
    budget_loss_weight: float = 0.1
    steps: int = 200
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 50
    seed: int = 42
    device: str = "auto"               # "auto" | "cpu" | "cuda"
    out_dir: str = "runs/debug"
    resume_from: Optional[str] = None

    # ---- GPU ----
    # bf16 只作用于**冻结的**大模型（encoder / generator）；Perceiver 压缩器始终留在 fp32，
    # 因为它是唯一在更新的部分，用低精度存优化器状态省不了多少却容易不稳。
    bf16: bool = False
    grad_accum: int = 1
    grad_ckpt: bool = False            # 对 hf_encoder 开梯度检查点（只在 LoRA 时才有意义）
    eval_max_samples: Optional[int] = None   # 评测抽样上限：贪心解码是逐条跑的，全量评很慢
    gen_max_new_tokens: int = 24
    # teacher_output 存在时优先作为 target，即序列级知识蒸馏。
    prefer_teacher_output: bool = True

    def __post_init__(self):
        assert self.stage == "stage1", "QuRO v0.0 only supports cache-first stage1 training"


@dataclass
class Config:
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    doc_encoder: DocEncoderConfig = field(default_factory=DocEncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        p, d, g = self.perceiver, self.doc_encoder, self.generator
        enc = d.kind + (f"({d.name_or_path}{',lora' if d.lora else ',frozen'})"
                        if d.kind == "hf_encoder" else "")
        return (
            f"[cfg] generator={g.name_or_path} | doc_encoder={enc} | "
            f"m={p.num_latents} d={p.d_latent} -> B_max={p.num_compressed} | "
            f"budgets={p.budget_buckets} adaptive={p.adaptive_budget} | "
            f"output_query={p.output_query_mode} | stage={self.train.stage} beta={self.train.beta_qa}"
        )


# --------------------------------------------------------------------------------------
# 预设
# --------------------------------------------------------------------------------------
def tiny_config() -> Config:
    """CPU 可跑：toy tokenizer + toy LM，只验证形状/梯度/loss 是否正常。"""
    cfg = Config()
    cfg.perceiver = PerceiverConfig(
        num_latents=64, d_latent=128,
        enc_num_heads=4, enc_num_blocks=1, enc_self_per_block=2,
        proc_num_layers=2, proc_num_heads=4,
        dec_num_heads=4, num_compressed=8,
        output_query_mode="xattn",
    )
    # 合成文档约 550 词 ≈ 700 词级 token，max_doc_len 必须够长，否则 gold 句可能被截掉
    cfg.doc_encoder = DocEncoderConfig(kind="standalone", d_model=128, max_doc_len=1024)
    cfg.generator = GeneratorConfig(
        name_or_path="toy", dtype="float32",
        toy_n_layer=4, toy_n_head=4, toy_d_model=256, toy_max_pos=1024,
    )
    cfg.data = DataConfig(
        train_file="data/multiq/train.jsonl",
        eval_files={"seen": "data/multiq/eval_seen.jsonl",
                    "unseen": "data/multiq/eval_unseen.jsonl"},
        max_doc_len=1024,
    )
    cfg.train = TrainConfig(steps=1500, batch_size=4, lr=3e-4, eval_every=250,
                            device="cpu", out_dir="runs/tiny")
    return cfg


def qwen7b_config() -> Config:
    """Cache-first readout with a frozen 7B backbone and generator LoRA."""
    cfg = Config()
    cfg.perceiver = PerceiverConfig(
        num_latents=256, d_latent=1024,
        enc_num_heads=8, enc_num_blocks=1, enc_self_per_block=2,
        proc_num_layers=6, proc_num_heads=8,
        dec_num_heads=8, num_compressed=8,
        output_query_mode="xattn",
    )
    cfg.doc_encoder = DocEncoderConfig(kind="generator_emb", max_doc_len=1024)
    cfg.generator = GeneratorConfig(
        name_or_path="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", freeze=True, lora=True,
    )
    cfg.data = DataConfig(
        train_file="data/multiq/train.jsonl",
        eval_files={"seen": "data/multiq/eval_seen.jsonl",
                    "unseen": "data/multiq/eval_unseen.jsonl"},
        max_doc_len=1024,
    )
    cfg.train = TrainConfig(steps=1000, batch_size=1, lr=1e-4, device="auto", out_dir="runs/qwen7b")
    return cfg


def qwen3emb_config() -> Config:
    """
    Main cache-first preset: frozen query encoder plus generator LoRA.

    三层分工（别混淆）：
        offline: frozen PISCO/COCOM -> per-document cache (m,h)
        online:  query tokens + retrieved K*m latents -> B readout embeddings
        consumer: frozen causal LM backbone + trainable LoRA
    """
    cfg = Config()
    cfg.perceiver = PerceiverConfig(
        num_latents=128, d_latent=768,
        enc_num_heads=12, enc_num_blocks=1, enc_self_per_block=2,
        proc_num_layers=6, proc_num_heads=12,
        dec_num_heads=12, num_compressed=8,
        output_query_mode="xattn",
    )
    cfg.doc_encoder = DocEncoderConfig(
        kind="hf_encoder", name_or_path=ENCODER_PATH, dtype="bfloat16",
        freeze=True, lora=False, max_doc_len=1024, add_learned_pos=False,
    )
    cfg.generator = GeneratorConfig(
        name_or_path=GENERATOR_PATH, dtype="bfloat16", freeze=True, lora=True)
    cfg.data = DataConfig(
        train_file=TRAIN_FILE, eval_files=dict(EVAL_FILES),
        fmt="auto", max_docs=1,
        max_doc_len=1024, max_query_len=64, max_answer_len=48,
    )
    cfg.train = TrainConfig(
        stage="stage1", steps=3000, batch_size=4, grad_accum=2, lr=1e-4,
        eval_every=500, device="auto", bf16=True, eval_max_samples=500,
        gen_max_new_tokens=32, out_dir="runs/qwen3emb",
    )
    return cfg


PRESETS = {
    "tiny": tiny_config,
    "qwen3emb": qwen3emb_config,
    "qwen7b": qwen7b_config,
}


def parse_eval_files(spec: str) -> Dict[str, str]:
    """解析 --eval_files "seen=a.jsonl,unseen=b.jsonl"。单独抽出来是为了不装 torch 也能测。"""
    out: Dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f'--eval_files 需要 name=path 形式，收到: "{chunk}"')
        name, path = chunk.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f'--eval_files 的 name 与 path 都不能为空: "{chunk}"')
        out[name] = path
    if not out:
        raise ValueError("--eval_files 解析后为空")
    return out


def get_config(preset: str = "tiny") -> Config:
    if preset not in PRESETS:
        raise KeyError(f"未知 preset={preset}，可选: {list(PRESETS)}")
    return PRESETS[preset]()
