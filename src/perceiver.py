"""
Perceiver IO 核心模块（按原论文 Jaegle et al., 2021 的结构重写，pre-LN + GEGLU FF）。

三个阶段：
    PerceiverEncoder   : cross-attn, Q = 可学习 latent 数组, K/V = 文档 token
                         >>> 这一步就是「全量压缩」：把任意长度 L 的文档压成固定 N 个 latent，
                         >>> query-agnostic，因此一篇文档只需编码一次、可跨 query 缓存复用。
                         >>> 对应 ICAE 的 memory slots / COCOM 的 context embeddings。
    PerceiverProcessor : latent 上的 self-attn 深加工，复杂度 O(N^2) 与文档长度无关。
    PerceiverDecoder   : cross-attn, Q = 由 RAG query 构造的 output query, K/V = latent
                         >>> 这一步是本工作的创新点：query 显式作为 Q 去 latent 里"提问"，
                         >>> 输出是 attention 的输出 embedding（不是 query 本身）。

注意力 mask 约定：全项目统一用 `mask` 表示 **key padding mask**，形状 (B, L_kv)，
bool，True = 有效 token，False = padding。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 基础组件
# ======================================================================================
class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """pre-LN + GEGLU FFN。"""

    def __init__(self, dim: int, widening: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * widening
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden * 2)   # *2 给 GEGLU 切两半
        self.act = GEGLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class MultiHeadAttention(nn.Module):
    """
    通用（cross / self）多头注意力，pre-LN 分别作用在 query 流和 key/value 流上。

    q_dim != kv_dim 是 Perceiver 的常态：
        Encode 时 q_dim = d_latent, kv_dim = d_doc；
        Decode 时 q_dim = d_latent(output query), kv_dim = d_latent。
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if head_dim is None:
            assert q_dim % num_heads == 0, f"q_dim={q_dim} 不能被 num_heads={num_heads} 整除，请显式给 head_dim"
            head_dim = q_dim // num_heads
        inner = num_heads * head_dim
        out_dim = out_dim or q_dim

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout_p = dropout

        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(q_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, out_dim)
        self.drop = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, N, Dh)

    def forward(
        self,
        x_q: torch.Tensor,                       # (B, Lq, q_dim)
        x_kv: torch.Tensor,                      # (B, Lkv, kv_dim)
        mask: Optional[torch.Tensor] = None,     # (B, Lkv) bool, True=有效
        return_attn: bool = False,
        score_bias: Optional[torch.Tensor] = None,   # (B, Lkv) or (B, Lq, Lkv)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``score_bias`` is added to the logits before the softmax.

        It exists so a readout can be *initialised* at a known-good scoring rule
        rather than at random.  A random ``to_q``/``to_k`` pair produces logits
        that are both tiny and semantically meaningless, and softmax over
        near-equal logits averages rather than selects -- measured at 99.85% of
        uniform entropy.  Feeding query-latent cosine similarity in here starts
        the attention at the non-parametric baseline's rule, leaving the learned
        term to act as a correction on top.

        An additive bias is used rather than an identity initialisation of
        ``to_q``/``to_k`` because the heads slice the feature dimension: with
        identity weights each head would score on its own 1/H of the dimensions,
        so the head-averaged map would not be the cosine at all.
        """
        q = self._split(self.to_q(self.q_norm(x_q)))
        kv = self.kv_norm(x_kv)
        k = self._split(self.to_k(kv))
        v = self._split(self.to_v(kv))

        bias = None
        if score_bias is not None:
            bias = score_bias[:, None, None, :] if score_bias.dim() == 2 else score_bias[:, None]
            bias = bias.to(q.dtype)

        attn_weights = None
        if return_attn or bias is not None:
            # 手写注意力：导出注意力图做可解释性可视化，以及施加打分偏置
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale       # (B, H, Lq, Lkv)
            if bias is not None:
                scores = scores + bias
            if mask is not None:
                scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
            attn_weights = scores.softmax(dim=-1)
            out = torch.matmul(F.dropout(attn_weights, self.dropout_p, self.training), v)
            if not return_attn:
                attn_weights = None
        else:
            attn_mask = mask[:, None, None, :] if mask is not None else None   # 自动广播到 (B,H,Lq,Lkv)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        b, h, lq, dh = out.shape
        out = out.transpose(1, 2).reshape(b, lq, h * dh)
        return self.drop(self.to_out(out)), attn_weights


class AttentionBlock(nn.Module):
    """attn + FFN，两个残差。cross / self 共用（self 时 x_kv = x_q）。"""

    def __init__(
        self,
        q_dim: int,
        kv_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        widening: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            q_dim=q_dim, kv_dim=kv_dim if kv_dim is not None else q_dim,
            num_heads=num_heads, head_dim=head_dim, out_dim=q_dim, dropout=dropout,
        )
        self.ff = FeedForward(q_dim, widening=widening, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        score_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kv = x if x_kv is None else x_kv
        a, w = self.attn(x, kv, mask=mask, return_attn=return_attn, score_bias=score_bias)
        x = x + a
        x = x + self.ff(x)
        return x, w


# ======================================================================================
# Encode：全量压缩（query-agnostic，可缓存）
# ======================================================================================
class PerceiverEncoder(nn.Module):
    """
    文档 (B, L, d_doc) --> latent (B, N, d_latent)。

    这是「全量压缩」阶段：不看 query，只把文档整体信息塞进 N 个 latent。
    与 ICAE/COCOM 的区别在于压缩由一次并行 cross-attention 完成，而非自回归生成 memory token。
    """

    def __init__(
        self,
        num_latents: int,
        d_latent: int,
        doc_dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        num_blocks: int = 1,
        self_per_block: int = 2,
        share_weights: bool = True,
        cross_widening: int = 1,
        self_widening: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.d_latent = d_latent
        self.num_blocks = num_blocks
        self.share_weights = share_weights

        # 可学习 latent 数组 —— Encode 阶段的 Q
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent) * 0.02)

        n_unique = 1 if share_weights else num_blocks
        self.cross_blocks = nn.ModuleList([
            AttentionBlock(q_dim=d_latent, kv_dim=doc_dim, num_heads=num_heads,
                           head_dim=head_dim, widening=cross_widening, dropout=dropout)
            for _ in range(n_unique)
        ])
        self.self_blocks = nn.ModuleList([
            nn.ModuleList([
                AttentionBlock(q_dim=d_latent, num_heads=num_heads, head_dim=head_dim,
                               widening=self_widening, dropout=dropout)
                for _ in range(self_per_block)
            ])
            for _ in range(n_unique)
        ])

    def forward(
        self,
        doc_emb: torch.Tensor,                    # (B, L, d_doc)
        doc_mask: Optional[torch.Tensor] = None,  # (B, L) bool
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b = doc_emb.size(0)
        x = self.latents.unsqueeze(0).expand(b, -1, -1)     # (B, N, d_latent)

        first_attn = None
        for i in range(self.num_blocks):
            idx = 0 if self.share_weights else i
            x, w = self.cross_blocks[idx](x, x_kv=doc_emb, mask=doc_mask,
                                          return_attn=return_attn and i == 0)
            if i == 0:
                first_attn = w                               # (B, H, N, L) latent 看了文档哪里
            for blk in self.self_blocks[idx]:
                x, _ = blk(x)
        return x, first_attn


# ======================================================================================
# Process：latent 深加工
# ======================================================================================
class PerceiverProcessor(nn.Module):
    def __init__(self, d_latent: int, num_layers: int = 6, num_heads: int = 8,
                 head_dim: Optional[int] = None, widening: int = 4, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(q_dim=d_latent, num_heads=num_heads, head_dim=head_dim,
                           widening=widening, dropout=dropout)
            for _ in range(num_layers)
        ])
        # Normalized prototype latents match the cache/readout interface scale.
        self.out_norm = nn.LayerNorm(d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x, _ = blk(x)
        return self.out_norm(x)


# ======================================================================================
# Output query 构造 —— 创新点的"提问方式"
# ======================================================================================
class OutputQueryBuilder(nn.Module):
    """
    把 RAG 的 query 变成 Decode 阶段的 P 个 output query（即 cross-attn 的 Q）。

    mode:
      agnostic : 只有 P 个可学习 slot，完全不用 query   -> 消融基线（"隐式"对照组）
      agnostic_matched
               : 与 xattn 同构、同参数量，但 K/V 换成固定长度的可学习占位序列，
                 与真实 query 无关。用于机制归因：此时 A 与 C 的唯一差别才真的是
                 "看不看 query"，而不是顺带少了一个 cross-attention block
                 （见 docs/warning_and_target.md W1 的参数量说明）。
      add      : slot + Linear(mean-pool(query tokens))
      film     : slot * (1 + scale(q)) + shift(q)
      concat   : Linear([slot; mean-pool(query)])，对应 e_q 与 P 的拼接
      xattn    : slot 先 cross-attend query 的 token 序列，每个 slot 可以关注 query 的不同部分
                 （对应 Option B/C 的多粒度 / 层次化设计）
    """

    #: Modes whose output does not depend on the query at all.
    QUERY_AGNOSTIC_MODES = ("agnostic", "agnostic_matched")

    def __init__(self, d_latent: int, num_compressed: int, query_dim: int,
                 mode: str = "xattn", num_heads: int = 8, dropout: float = 0.0,
                 placeholder_len: int = 16):
        super().__init__()
        self.mode = mode
        self.num_compressed = num_compressed
        self.slots = nn.Parameter(torch.randn(num_compressed, d_latent) * 0.02)

        if mode == "add":
            self.proj = nn.Linear(query_dim, d_latent)
        elif mode == "film":
            self.proj = nn.Linear(query_dim, d_latent * 2)
        elif mode == "concat":
            self.proj = nn.Linear(query_dim + d_latent, d_latent)
        elif mode in ("xattn", "agnostic_matched"):
            self.block = AttentionBlock(q_dim=d_latent, kv_dim=query_dim, num_heads=num_heads,
                                        widening=1, dropout=dropout)
        if mode == "agnostic_matched":
            # Fixed length, so it cannot leak the real query's token count either.
            self.placeholder = nn.Parameter(torch.randn(placeholder_len, query_dim) * 0.02)
        self.norm = nn.LayerNorm(d_latent)

    @property
    def is_query_agnostic(self) -> bool:
        return self.mode in self.QUERY_AGNOSTIC_MODES

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def forward(
        self,
        query_emb: Optional[torch.Tensor] = None,    # (B, Lq, query_dim)
        query_mask: Optional[torch.Tensor] = None,   # (B, Lq)
        batch_size: Optional[int] = None,
        num_outputs: Optional[int] = None,
    ) -> torch.Tensor:
        p = self.num_compressed if num_outputs is None else int(num_outputs)
        if not 1 <= p <= self.num_compressed:
            raise ValueError(f"num_outputs must be in [1,{self.num_compressed}], got {p}")
        base_slots = self.slots[:p]
        if self.is_query_agnostic:
            assert batch_size is not None or query_emb is not None
            b = batch_size if query_emb is None else query_emb.size(0)
            slots = base_slots.unsqueeze(0).expand(b, -1, -1)
            if self.mode == "agnostic":
                return self.norm(slots)
            # agnostic_matched: same block, same cost, query-independent K/V.
            kv = self.placeholder.unsqueeze(0).expand(b, -1, -1)
            out, _ = self.block(slots, x_kv=kv, mask=None)
            return self.norm(out)

        assert query_emb is not None, f"output_query_mode={self.mode} 需要 query_emb"
        b = query_emb.size(0)
        slots = base_slots.unsqueeze(0).expand(b, -1, -1)          # (B, P, d)

        if self.mode == "add":
            q = self.proj(self._masked_mean(query_emb, query_mask))  # (B, d)
            out = slots + q.unsqueeze(1)
        elif self.mode == "film":
            scale, shift = self.proj(self._masked_mean(query_emb, query_mask)).chunk(2, dim=-1)
            out = slots * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        elif self.mode == "concat":
            q = self._masked_mean(query_emb, query_mask).unsqueeze(1).expand(-1, p, -1)
            out = self.proj(torch.cat([slots, q], dim=-1))
        else:  # xattn
            out, _ = self.block(slots, x_kv=query_emb, mask=query_mask)
        return self.norm(out)


# ======================================================================================
# Decode：query-as-Q（创新点本体）
# ======================================================================================
class PerceiverDecoder(nn.Module):
    """
    output_query (B, P, d) 作为 Q，latent (B, N, d) 作为 K/V，
    一次并行 cross-attention 直接吐出 P 个压缩 embedding。

    对比 SeleCom：它自回归地逐个生成 p 个 special token；这里是 O(1) 步的并行抽取。
    """

    def __init__(self, d_latent: int, num_heads: int = 8, head_dim: Optional[int] = None,
                 widening: int = 1, dropout: float = 0.0):
        super().__init__()
        self.block = AttentionBlock(q_dim=d_latent, kv_dim=d_latent, num_heads=num_heads,
                                    head_dim=head_dim, widening=widening, dropout=dropout)
        self.out_norm = nn.LayerNorm(d_latent)

    def forward(self, output_query: torch.Tensor, latents: torch.Tensor,
                latent_mask: Optional[torch.Tensor] = None,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, w = self.block(output_query, x_kv=latents, mask=latent_mask,
                          return_attn=return_attn)
        return self.out_norm(x), w     # w: (B, H, P, N)，可视化"第 i 个压缩位关注了哪些 latent"


# ======================================================================================
# Projector：latent 空间 -> generator 表示空间
# ======================================================================================
class Projector(nn.Module):
    def __init__(self, d_in: int, d_out: int, kind: str = "mlp",
                 hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(d_in, d_out)
        else:
            hidden = d_in * hidden_mult
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, d_out),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
