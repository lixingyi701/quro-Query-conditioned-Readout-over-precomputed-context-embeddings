"""SeleCom's "full compression is infeasible" claim, measured on PISCO.

SeleCom's Figure 2 reports that a generator reading full-compression embeddings
keeps reciting the document even when the instruction says to ignore it, and that
its attention stays pinned on the compressed positions.  This module is the
measurement apparatus for asking whether the same thing happens to *our* PISCO
baseline -- the ``P`` arm, unchanged -- rather than for drawing a similar picture.

Three things have to be right for the answer to mean anything, and each one has a
section below.

``render``/``token_groups``
    The decoder's input is one flat sequence of embeddings, so "attention on the
    memory" is only defined once every position is assigned to a named group.
    Guessing index ranges from string lengths is how that goes wrong silently, so
    the groups are carried as *character* spans through the chat template and
    converted to token spans with the fast tokenizer's offset mapping, with the
    boundary-lands-on-a-token-boundary condition checked rather than assumed.
    The rendering is also a strict generalisation of :class:`~src.prompt.PiscoPromptBuilder`:
    with an empty instruction it reproduces ``D0`` (slots) and ``RG`` (raw text)
    token for token, which ``tests/test_attention_grouping.py`` asserts.

``CollectedAttention``/``capture_attention``
    Attention *mass* on a group grows with the number of tokens in it, and PISCO
    memory at K=10 is 80 positions against an instruction's ~20.  Mass alone would
    therefore "confirm" SeleCom on token count alone, so mass and density are
    always produced together, alongside the pre-softmax QK logits and the K/V
    norms that SeleCom's Appendix A.1 blames -- those are the quantities that tell
    a length effect apart from a scale effect.  Collection happens inside a patched
    ``eager_attention_forward`` and aggregates per layer, so the full
    ``[layers, heads, T, T]`` tensor is never held.

``score_conflict``/``score_reconstruction``
    Behaviour first, mechanism second (instruction §2).  An attention finding with
    no behavioural failure under it is not evidence of anything.

Nothing here trains, and nothing here is a proposed fix.
"""

from __future__ import annotations

import math
import random
import re
import string
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .prompt import SYSTEM_PROMPT

# --------------------------------------------------------------------------------------
# 1. Prompt rendering with auditable group spans
# --------------------------------------------------------------------------------------

#: Ordered, contiguous and exhaustive partition of the decoder's input positions.
#: ``document`` covers whichever representation the condition uses -- PISCO memory
#: slots or raw text -- so the mass/density ratios are directly comparable across
#: the 2x2; the condition records *which* under ``document_kind``.
GROUPS: Tuple[str, ...] = (
    "prefix",              # BOS + [INST] + system prompt
    "document_delimiter",  # "Background:\n"
    "document",            # <MEM*> slots, raw document tokens, or empty
    "query_delimiter",     # "\n\nQuestion:"
    "query",               # the question, empty in the Level A reconstruction/conflict pair
    "instruction",         # the conflicting instruction, empty in a plain QA row
    "answer_prefix",       # " [/INST]"
    "output_history",      # teacher-forced target tokens; empty in the prompt itself
)
GROUP_INDEX: Dict[str, int] = {name: i for i, name in enumerate(GROUPS)}


@contextmanager
def _quiet_nan():
    """Silence the all-NaN reductions that absent groups legitimately produce.

    A condition without a document (the no-memory control) or without a query
    (Level A) really has no values to average, and NaN is the right answer -- but
    numpy warns once per layer per row, which would bury every other message.
    """
    import warnings

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield

BACKGROUND_MARKER = "Background:\n"
QUESTION_MARKER = "\n\nQuestion:"

#: How the instruction is introduced.  ``pisco_question`` puts it where PISCO's
#: decoder was trained to find the question, which keeps the prompt in
#: distribution and makes the rendering token-identical to ``D0``/``RG``;
#: ``selecom_literal`` drops the ``Question:`` marker so the instruction reads as
#: SeleCom wrote it.  Both exist because a null result under one of them could
#: otherwise be an artefact of the scaffolding rather than a statement about
#: compression -- the marker tells a RAG-tuned decoder that it is being asked a
#: question about the background, which is itself a pull towards the document.
PROMPT_STYLES = ("pisco_question", "selecom_literal")


@dataclass
class RenderedPrompt:
    """A prompt string plus the character span of every group inside it."""

    text: str
    char_spans: Dict[str, Tuple[int, int]]

    def check(self) -> None:
        """Spans must tile ``text`` exactly, in ``GROUPS`` order."""
        cursor = 0
        for name in GROUPS:
            if name == "output_history" or name not in self.char_spans:
                continue
            start, end = self.char_spans[name]
            if start != cursor:
                raise ValueError(
                    f"group {name!r} starts at {start} but the previous group ended "
                    f"at {cursor}; the rendered prompt is not tiled by its groups")
            if end < start:
                raise ValueError(f"group {name!r} has a negative span {(start, end)}")
            cursor = end
        if cursor != len(self.text):
            raise ValueError(
                f"groups cover {cursor} of {len(self.text)} characters; "
                "the tail of the prompt is unassigned")


def render(tokenizer, document: Optional[str], query: str = "", instruction: str = "",
           system_prompt: Optional[str] = SYSTEM_PROMPT,
           style: str = "pisco_question") -> RenderedPrompt:
    """Render PISCO's chat prompt and report where each group landed.

    ``document`` is already-rendered *text*: the ``<MEM0>..<MEM7><SEP>`` slot
    string for a compressed condition, the raw passage(s) for the uncompressed
    one, and ``None`` for the no-memory control -- which drops the whole
    ``Background:`` block, matching the repository's existing ``AG`` mode rather
    than leaving a visibly empty heading behind.

    ``query`` and ``instruction`` are concatenated in that order and kept as
    separate groups, because SeleCom's claim is about how the two compete for
    attention and merging them would make the central ratio undefined.

    ``system_prompt=None`` removes the system turn entirely.  PISCO's is not
    neutral -- it tells the model its task is to extract from the provided
    documents -- so a conflict instruction is arguing with a standing order, and
    a result obtained only under it would not separate "compression captures
    attention" from "the system prompt said to read the documents".
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected {PROMPT_STYLES}")
    marker = QUESTION_MARKER if style == "pisco_question" else "\n\n"

    user_parts: List[Tuple[str, str]] = []
    if document is not None:
        user_parts.append(("document_delimiter", BACKGROUND_MARKER))
        user_parts.append(("document", document))
        user_parts.append(("query_delimiter", marker))
    else:
        user_parts.append(("query_delimiter", marker.lstrip("\n")))
    user_parts.append(("query", query))
    user_parts.append(("instruction", instruction))

    user = "".join(text for _, text in user_parts)
    if not user.strip():
        raise ValueError("the user turn is empty; there is nothing to prompt with")
    # PiscoPromptBuilder._chat applies this substitution before templating; it is a
    # no-op for every string built here, and a length-changing edit would silently
    # shift every span, so refuse rather than re-derive the offsets afterwards.
    if ":\\ " in user:
        raise ValueError("the rendered user turn contains ':\\ ', which the PISCO "
                         "prompt builder rewrites; the group spans would shift")

    if system_prompt is None:
        text = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user}]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        except Exception:
            # Mistral-v0.2 has no system role; PISCO folds it into the user turn.
            merged = [{"role": "user", "content": system_prompt + "\n" + user}]
            text = tokenizer.apply_chat_template(merged, tokenize=False,
                                                 add_generation_prompt=True)

    offset = text.find(user)
    if offset < 0:
        raise ValueError("the chat template altered the user turn; group spans "
                         "cannot be located in the rendered prompt")

    spans: Dict[str, Tuple[int, int]] = {}
    cursor = offset
    spans["prefix"] = (0, offset)
    for name, piece in user_parts:
        spans[name] = (cursor, cursor + len(piece))
        cursor += len(piece)
    # Everything the template appended after the user turn is the answer prefix.
    spans["answer_prefix"] = (cursor, len(text))

    rendered = RenderedPrompt(text=text, char_spans=spans)
    rendered.check()
    return rendered


@dataclass
class TokenGroups:
    """Token ids for one prompt, plus ``[start, end)`` per group."""

    input_ids: List[int]
    spans: Dict[str, Tuple[int, int]]
    text: str
    #: Tokens whose characters cross a group boundary -- always whitespace-only
    #: crossings, since anything else is refused.  Recorded so the manifest can
    #: show the assignment was exact rather than asserting it.
    straddling_tokens: List[int] = field(default_factory=list)

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.input_ids)

    def sizes(self) -> Dict[str, int]:
        return {name: end - start for name, (start, end) in self.spans.items()}

    def with_targets(self, target_ids: Sequence[int]) -> "TokenGroups":
        """Append teacher-forced targets as the ``output_history`` group."""
        start = len(self.input_ids)
        spans = dict(self.spans)
        spans["output_history"] = (start, start + len(target_ids))
        return TokenGroups(list(self.input_ids) + list(target_ids), spans, self.text,
                           list(self.straddling_tokens))

    def index_tensor(self, total_length: Optional[int] = None) -> torch.Tensor:
        """``(T,)`` group id per position; ``-1`` for anything unassigned."""
        total = total_length if total_length is not None else len(self.input_ids)
        index = torch.full((total,), -1, dtype=torch.long)
        for name, (start, end) in self.spans.items():
            index[start:end] = GROUP_INDEX[name]
        return index

    def check(self) -> None:
        """Non-overlapping, gap-free, and exactly as long as the token sequence."""
        covered = torch.zeros(len(self.input_ids), dtype=torch.long)
        for name, (start, end) in self.spans.items():
            if name == "output_history":
                continue
            if start < 0 or end > len(self.input_ids) or end < start:
                raise ValueError(f"group {name!r} span {(start, end)} is out of range "
                                 f"for {len(self.input_ids)} tokens")
            covered[start:end] += 1
        if int(covered.max()) > 1:
            overlapping = [i for i, c in enumerate(covered.tolist()) if c > 1]
            raise ValueError(f"positions {overlapping[:8]} belong to more than one group")
        if int(covered.min()) < 1:
            missing = [i for i, c in enumerate(covered.tolist()) if c < 1]
            raise ValueError(f"positions {missing[:8]} belong to no group")


def token_groups(tokenizer, rendered: RenderedPrompt,
                 max_prompt_tokens: int = 4096) -> TokenGroups:
    """Convert character spans to token spans through the offset mapping.

    Each *token* is assigned to the group its first **non-whitespace** character
    falls in.  That makes the result a partition by construction -- contiguous,
    gap-free and exhaustive -- rather than something to be checked afterwards,
    and it is why the same code works whether or not a group boundary happens to
    coincide with a token boundary.

    The awkward case is real and not hypothetical: the chat template emits
    ``[INST] Background:``, and SentencePiece merges the preceding space into
    ``▁Background``, so the prefix/delimiter boundary lands mid-token.  Keying on
    the first non-whitespace character puts that token with the word it spells
    rather than with the space in front of it; keying on the first character
    would file ``Background`` under ``prefix``.  A token whose *content*
    characters fall in two different groups would cost real text either way, so
    that is an error.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer is required: the group spans come from "
                         "its offset mapping, not from string arithmetic")
    encoded = tokenizer(rendered.text, add_special_tokens=False,
                        return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if len(input_ids) > max_prompt_tokens:
        raise ValueError(
            f"prompt is {len(input_ids)} tokens, over the {max_prompt_tokens} cap; "
            "truncating would cut the instruction, which sits last")

    present = [name for name in GROUPS
               if name != "output_history" and name in rendered.char_spans]
    bounds = [rendered.char_spans[name] for name in present]

    def group_of(char_index: int) -> int:
        index = 0
        for i, (a, b) in enumerate(bounds):
            if a <= char_index < b or (a == b == char_index) or char_index >= b:
                index = i
        return index

    owner: List[int] = []
    for start, stop in offsets:
        content = next((c for c in range(start, stop)
                        if not rendered.text[c].isspace()), start)
        owner.append(group_of(content))

    spans: Dict[str, Tuple[int, int]] = {}
    for i, name in enumerate(present):
        members = [t for t, g in enumerate(owner) if g == i]
        if members:
            spans[name] = (members[0], members[-1] + 1)
        else:
            # A group with no token of its own -- an empty query, or one whose
            # characters were swallowed by a neighbour's token.  It still needs a
            # position, or the manifest cannot say where it would have been.
            anchor = next((t for t, (a, _) in enumerate(offsets)
                           if a >= bounds[i][0]), len(offsets))
            spans[name] = (anchor, anchor)

    straddling = []
    for t, (a, b) in enumerate(offsets):
        touched = {group_of(c) for c in range(a, b)}
        if len(touched) < 2:
            continue
        content = {group_of(c) for c in range(a, b) if not rendered.text[c].isspace()}
        if len(content) > 1:
            raise ValueError(
                f"token {t} ({rendered.text[a:b]!r}) spans the groups "
                f"{sorted(present[i] for i in content)} with content in each; "
                "assigning it to one of them would move real text between groups")
        straddling.append(t)

    groups = TokenGroups(input_ids=input_ids, spans=spans, text=rendered.text,
                         straddling_tokens=straddling)
    groups.check()
    return groups


def slot_string(tokenizer, n_latents: int, n_mem_tokens: int) -> str:
    """``n_latents`` memory slots laid out as PISCO's blocks plus ``<SEP>``.

    Same layout as :meth:`src.prompt.PiscoPromptBuilder.slot_string`; duplicated
    here only because this module renders the prompt around it, and the
    equivalence is asserted by the tests rather than assumed.
    """
    if n_latents < 1:
        raise ValueError("n_latents must be >= 1")
    mem_tokens = list(getattr(tokenizer, "mem_tokens", []))
    if not mem_tokens:
        raise ValueError("tokenizer exposes no mem_tokens")
    sep = getattr(tokenizer, "sep_token", "") or ""
    out, remaining = [], n_latents
    while remaining > 0:
        take = min(n_mem_tokens, remaining)
        out.append("".join(mem_tokens[:take]) + sep)
        remaining -= take
    return "".join(out)


def memory_positions(tokenizer, groups: TokenGroups) -> List[int]:
    """Positions holding ``<MEM*>`` tokens, which must lie inside ``document``."""
    mem_ids = set(getattr(tokenizer, "mem_token_ids", []) or [])
    if not mem_ids:
        return []
    found = [i for i, token in enumerate(groups.input_ids) if token in mem_ids]
    start, end = groups.spans.get("document", (0, 0))
    outside = [i for i in found if not (start <= i < end)]
    if outside:
        raise ValueError(f"memory tokens at {outside[:8]} fall outside the document "
                         f"group {(start, end)}")
    return found


# --------------------------------------------------------------------------------------
# 2. Conditions
# --------------------------------------------------------------------------------------

#: How the document reaches the decoder.  ``mismatch`` and ``none``/``zero`` are
#: the qualification gate from instruction §4: without a clear correct-vs-mismatch
#: gap the PISCO memory is not carrying the document, and "it suppressed the
#: instruction" is not a statement the data can support.
DOCUMENT_KINDS = ("memory", "mismatch", "raw", "none", "zero")

#: What the decoder is asked to do.
TASKS = ("reconstruct", "conflict", "qa", "qa_conflict")

RECONSTRUCTION_INSTRUCTION = "Verbatim the background again (including title and content)."
CONFLICT_TEMPLATE = ('Ignore the above content and output exactly the following '
                     'string "{nonce}".')


#: Inference-time, reversible edits to the memory embeddings (instruction §11).
#: These are diagnostics, not proposals: each one removes a different property of
#: the PISCO latents while leaving the slots, the positions and the prompt alone,
#: so "which property of the vectors does the work" becomes a measurement.
#:
#: ``scale:<a>``            a * Z -- interpolates the latents towards the zero
#:                          control, which already behaves like raw text
#: ``norm_matched_random``  random directions at the real per-slot norms -- keeps
#:                          the scale, discards the direction and the subspace
#: ``shuffle``              permutes the slot order -- keeps every vector, breaks
#:                          the sequence (an order test only; the information is
#:                          NOT preserved in any stronger sense)
#: ``mean``                 every slot gets the row's mean latent -- keeps the
#:                          distribution's location, removes per-slot content
#:
#: The corpus-level family below tests whether the suppressing component is
#: *separable* from the document.  ``mean`` already showed that the row mean
#: carries the suppression while the per-slot deviations carry the document, so
#: the question is whether a single corpus-wide offset does the same job:
#:
#: ``global_mean``          every slot gets the corpus mean latent -- carries no
#:                          document at all, not even this row's
#: ``decenter``             Z - mu, which also shrinks the norm
#: ``decenter_renorm``      Z - mu rescaled back to ||Z||, so the direction is
#:                          removed with the scale held fixed (the scale sweep
#:                          already showed scale is inert, but conflating the two
#:                          would make the result unreadable)
#: ``deproject:<k>``        remove the top-k principal directions of the latent
#:                          distribution, renormalised
TRANSFORMS = ("none", "norm_matched_random", "shuffle", "mean",
              "global_mean", "decenter", "decenter_renorm")

#: Transforms that need corpus-level statistics rather than just this row's.
CORPUS_TRANSFORMS = ("global_mean", "decenter", "decenter_renorm", "deproject")


def parse_transform(spec: str) -> Tuple[str, float]:
    for prefix in ("scale", "deproject"):
        if spec.startswith(prefix + ":"):
            return prefix, float(spec.split(":", 1)[1])
    if spec not in TRANSFORMS:
        raise ValueError(f"unknown memory transform {spec!r}; expected one of "
                         f"{TRANSFORMS} or 'scale:<alpha>' / 'deproject:<k>'")
    return spec, float("nan")


@dataclass
class LatentStatistics:
    """Corpus-level geometry of the cached latents.

    Estimated once over a sample of the cache and stored with the run, so a
    subspace transform is a fixed function of the corpus rather than of whatever
    documents happened to be in the batch.
    """

    mean: torch.Tensor                 # (h,)
    basis: torch.Tensor                # (k, h), orthonormal, top-k directions
    singular_values: torch.Tensor      # (k,)
    n_documents: int
    n_vectors: int
    seed: int

    def summary(self) -> Dict[str, object]:
        norms = self.singular_values
        return {"n_documents": self.n_documents, "n_vectors": self.n_vectors,
                "seed": self.seed, "mean_norm": round(float(self.mean.norm()), 4),
                "basis_rank": int(self.basis.size(0)),
                "top_singular_values": [round(float(v), 2) for v in norms[:8]],
                # How much of the raw second moment the leading directions hold.
                "energy_fraction_top1": round(float((norms[0] ** 2) / (norms ** 2).sum()), 4),
                "energy_fraction_top8": round(
                    float((norms[:8] ** 2).sum() / (norms ** 2).sum()), 4)}

    def to_device(self, device, dtype) -> "LatentStatistics":
        return LatentStatistics(
            self.mean.to(device=device, dtype=dtype),
            self.basis.to(device=device, dtype=dtype),
            self.singular_values, self.n_documents, self.n_vectors, self.seed)


@torch.no_grad()
def estimate_latent_statistics(cache, n_documents: int = 8192, rank: int = 32,
                               seed: int = 0, device="cpu") -> LatentStatistics:
    """Mean and leading directions of the cached latent distribution.

    The sample is over *documents*, not over the queries that happen to retrieve
    them, so the estimate describes the cache and not the eval set.
    """
    doc_ids = sorted(cache.documents)
    rng = np.random.default_rng(seed)
    if len(doc_ids) > n_documents:
        picked = [doc_ids[i] for i in rng.choice(len(doc_ids), n_documents, replace=False)]
    else:
        picked = doc_ids
    latents, _, _ = cache.get_many([[d] for d in picked])
    flat = latents.reshape(-1, latents.size(-1)).float().to(device)
    mean = flat.mean(0)
    # Uncentered SVD: direction 1 is then essentially mu/||mu||, which is what the
    # decenter transform removes -- keeping the two comparable.
    _, singular, right = torch.svd_lowrank(flat, q=min(rank + 8, flat.size(0) - 1))
    return LatentStatistics(mean=mean.cpu(), basis=right[:, :rank].T.contiguous().cpu(),
                            singular_values=singular[:rank].cpu(),
                            n_documents=len(picked), n_vectors=int(flat.size(0)), seed=seed)


def apply_transform(soft: torch.Tensor, spec: str,
                    generator: Optional[torch.Generator] = None,
                    stats: Optional[LatentStatistics] = None) -> torch.Tensor:
    """Edit the memory embeddings and nothing else."""
    kind, value = parse_transform(spec)
    if kind == "none":
        return soft
    if kind == "scale":
        return soft * value
    if kind == "mean":
        return soft.mean(0, keepdim=True).expand_as(soft).contiguous()
    if kind == "shuffle":
        order = torch.randperm(soft.size(0), generator=generator,
                               device="cpu").to(soft.device)
        return soft.index_select(0, order)
    if kind in CORPUS_TRANSFORMS:
        if stats is None:
            raise ValueError(f"{spec!r} needs corpus latent statistics")
        stats = stats.to_device(soft.device, torch.float32)
        work = soft.float()
        if kind == "global_mean":
            return stats.mean[None].expand_as(work).to(soft.dtype).contiguous()
        norms = work.norm(dim=-1, keepdim=True)
        if kind == "deproject":
            basis = stats.basis[: int(value)]                  # (k, h)
            work = work - (work @ basis.T) @ basis
        else:
            work = work - stats.mean[None]
        if kind in ("decenter_renorm", "deproject"):
            work = work * (norms / work.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return work.to(soft.dtype)
    # norm_matched_random: same per-slot norm, a direction drawn from nowhere in
    # particular.  If this suppresses the instruction as much as the real latents
    # do, the cause is the scale and the out-of-distribution-ness, not the
    # compressed document.
    noise = torch.randn(soft.shape, generator=generator, device="cpu",
                        dtype=torch.float32).to(soft.device)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (noise * soft.float().norm(dim=-1, keepdim=True)).to(soft.dtype)


@dataclass
class Condition:
    """One cell of the experiment matrix."""

    name: str
    document_kind: str
    task: str
    #: Inference-time edit applied to the memory embeddings; ``none`` for every
    #: Phase 1-2 condition, so the observational results carry no intervention.
    transform: str = "none"

    def __post_init__(self):
        if self.document_kind not in DOCUMENT_KINDS:
            raise ValueError(f"unknown document_kind {self.document_kind!r}")
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}")
        parse_transform(self.transform)
        if self.transform != "none" and self.document_kind != "memory":
            raise ValueError("a memory transform only applies to the memory condition")

    @property
    def compressed(self) -> bool:
        return self.document_kind in ("memory", "mismatch", "zero")


def level_a_conditions() -> List[Condition]:
    """SeleCom Figure 2's 2x2, plus the controls §6 requires alongside it."""
    return [
        Condition("memory/reconstruct", "memory", "reconstruct"),
        Condition("memory/conflict", "memory", "conflict"),
        Condition("raw/reconstruct", "raw", "reconstruct"),
        Condition("raw/conflict", "raw", "conflict"),
        Condition("mismatch/reconstruct", "mismatch", "reconstruct"),
        Condition("mismatch/conflict", "mismatch", "conflict"),
        Condition("none/reconstruct", "none", "reconstruct"),
        Condition("none/conflict", "none", "conflict"),
        Condition("zero/conflict", "zero", "conflict"),
    ]


def intervention_conditions(alphas: Sequence[float] = (0.25, 0.5, 0.75)) -> List[Condition]:
    """Phase 3: which property of the latents drives the failure? (§11)

    Both ends are always measured (§11's closing requirement): every transform is
    run under the conflict instruction *and* under reconstruction, so an edit that
    restores instruction following by destroying the document shows up as such
    rather than as a fix.

    The observational anchors -- untouched memory, raw text, no memory -- are in
    the same run so the intervention is read against its own controls rather than
    against a different sweep's.
    """
    out = [Condition("memory/conflict", "memory", "conflict"),
           Condition("memory/reconstruct", "memory", "reconstruct"),
           Condition("raw/conflict", "raw", "conflict"),
           Condition("raw/reconstruct", "raw", "reconstruct"),
           Condition("zero/conflict", "zero", "conflict"),
           Condition("none/conflict", "none", "conflict")]
    for alpha in alphas:
        spec = f"scale:{alpha}"
        out.append(Condition(f"memory@a{alpha}/conflict", "memory", "conflict", spec))
        out.append(Condition(f"memory@a{alpha}/reconstruct", "memory", "reconstruct", spec))
    for name in ("norm_matched_random", "shuffle", "mean"):
        tag = {"norm_matched_random": "rand", "shuffle": "shuf", "mean": "mean"}[name]
        out.append(Condition(f"memory@{tag}/conflict", "memory", "conflict", name))
        out.append(Condition(f"memory@{tag}/reconstruct", "memory", "reconstruct", name))
    return out


def subspace_conditions(ranks: Sequence[int] = (1, 2, 4, 8, 16)) -> List[Condition]:
    """Is the suppressing component separable from the document? (innovation #2)

    ``mean`` established the decomposition ``Z_i = mu_row + delta_i``: replacing
    every slot by the row mean keeps the suppression in full and halves the
    document.  So the suppression rides on a location and the document rides on
    the deviations -- which makes "remove the location, keep the deviations" a
    question with a yes/no answer rather than a design.

    Both directions are measured.  ``global_mean`` is the complement: a single
    corpus-wide vector carrying no document at all.  If *that* suppresses, the
    suppressing component is not even row-specific.
    """
    out = [Condition("memory/conflict", "memory", "conflict"),
           Condition("memory/reconstruct", "memory", "reconstruct"),
           Condition("raw/conflict", "raw", "conflict"),
           Condition("raw/reconstruct", "raw", "reconstruct"),
           Condition("zero/conflict", "zero", "conflict"),
           Condition("none/conflict", "none", "conflict"),
           Condition("none/reconstruct", "none", "reconstruct")]
    for name, tag in (("global_mean", "gmean"), ("decenter", "dec"),
                      ("decenter_renorm", "decn"), ("mean", "rmean")):
        out.append(Condition(f"memory@{tag}/conflict", "memory", "conflict", name))
        out.append(Condition(f"memory@{tag}/reconstruct", "memory", "reconstruct", name))
    for k in ranks:
        spec = f"deproject:{k}"
        out.append(Condition(f"memory@proj{k}/conflict", "memory", "conflict", spec))
        out.append(Condition(f"memory@proj{k}/reconstruct", "memory", "reconstruct", spec))
    return out


def level_b_alpha_conditions(alphas: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.50)
                             ) -> List[Condition]:
    """The scale sweep on the metric that decides the project (innovation #1 stage B).

    Level A measures document utility by reconstruction, which is the task PISCO
    was distilled on and therefore the one most favourable to the latents.  The
    number QuRO is judged on is HotpotQA QA, and the gap there lives entirely in
    the bridge questions -- so the sweep has to run on QA, not on reconstruction,
    or it would pick a scale by optimising the wrong quantity.

    Deliberately small: the point is the alpha-by-contextualisation-by-accuracy
    curve, and every extra condition costs 200 rows of generation.
    """
    out = [Condition("raw/qa", "raw", "qa"),
           Condition("memory/qa", "memory", "qa"),
           Condition("mismatch/qa", "mismatch", "qa"),
           Condition("none/qa", "none", "qa")]
    for alpha in alphas:
        out.append(Condition(f"memory@a{alpha}/qa", "memory", "qa", f"scale:{alpha}"))
    return out


def level_b_conditions() -> List[Condition]:
    """HotpotQA: the same contrast where the repository's numbers actually live."""
    return [
        Condition("memory/qa", "memory", "qa"),
        Condition("memory/qa_conflict", "memory", "qa_conflict"),
        Condition("memory/conflict", "memory", "conflict"),
        Condition("raw/qa", "raw", "qa"),
        Condition("raw/qa_conflict", "raw", "qa_conflict"),
        Condition("raw/conflict", "raw", "conflict"),
        Condition("mismatch/qa", "mismatch", "qa"),
        Condition("none/qa", "none", "qa"),
        Condition("none/conflict", "none", "conflict"),
        Condition("zero/conflict", "zero", "conflict"),
    ]


# --------------------------------------------------------------------------------------
# 3. Nonce strings
# --------------------------------------------------------------------------------------

_NONCE_ALPHABET = string.ascii_uppercase


def make_nonce(rng: random.Random, tokenizer, length: int = 12,
               max_attempts: int = 64) -> Dict[str, object]:
    """A per-example random target string that survives a tokenise/decode round trip.

    Per-example rather than one constant for the whole set: a single fixed string
    would let "the model emits it" be memorisation of the run rather than
    instruction following, and would make the exact-match rate a property of one
    draw.  Uppercase letters follow SeleCom's own example; the round-trip check is
    what makes the token-level metric well defined.
    """
    for _ in range(max_attempts):
        nonce = "".join(rng.choice(_NONCE_ALPHABET) for _ in range(length))
        ids = tokenizer(nonce, add_special_tokens=False)["input_ids"]
        if tokenizer.decode(ids, skip_special_tokens=True) == nonce:
            return {"nonce": nonce, "nonce_token_ids": list(ids),
                    "nonce_n_tokens": len(ids)}
    raise RuntimeError(f"no {length}-character nonce round-tripped through the "
                       f"tokenizer in {max_attempts} attempts")


# --------------------------------------------------------------------------------------
# 4. Grouped attention, norms and pre-softmax logits
# --------------------------------------------------------------------------------------

@dataclass
class CollectedAttention:
    """Per-layer statistics, aggregated online so no full attention tensor is kept.

    Shapes, with ``L`` layers, ``H`` heads, ``S`` teacher-forced target positions
    and ``G = len(GROUPS)``:

    ``mass``        ``[L, H, S, G]``  post-softmax attention summed over a group
    ``group_size``  ``[S, G]``        causally visible tokens per group per target
    ``qk_mean``     ``[L, H, S, G]``  pre-softmax QK logits, averaged over the group
    ``qk_max``      ``[L, H, S, G]``  the strongest single pre-softmax logit
    ``k_norm``      ``[L, H, G]``     mean ``||k_j||`` over the group
    ``v_norm``      ``[L, H, G]``     mean ``||v_j||`` over the group
    ``v_contrib``   ``[L, H, S, G]``  ``||sum_{j in g} A_tj v_j||``, the group's
                                      actual contribution to the attention output
    ``entropy``     ``[L, H, S]``     over the full source axis
    ``hidden_norm`` ``[L+1, G]``      mean residual-stream norm per group
    ``top_group``   ``[L, H, S]``     group id of the single most attended position
    ``hidden_update``   ``[L, G]``    ``||h_{l+1} - h_l|| / ||h_l||`` per group
    ``hidden_rotation`` ``[L, G]``    ``cos(h_{l+1}, h_l)`` per group

    The last two are what decide whether a memory position is *processed* by the
    decoder or merely read from it.  ``hidden_norm`` alone cannot say: in a
    pre-norm stack the per-layer update is computed from ``RMSNorm(h)``, so it has
    a magnitude set by the layer rather than by ``||h||`` -- add an O(10) update
    to a text token at norm 0.14 and it is rewritten, add the same update to a
    memory slot at norm 104 and almost nothing happens.  The ratio and the
    rotation measure that directly.

    ``mass`` is deliberately not reduced over heads or targets here: §8.5 asks for
    head-level medians and the fraction of memory-dominant heads, and a mean taken
    too early cannot be un-taken.
    """

    mass: np.ndarray
    group_size: np.ndarray
    qk_mean: np.ndarray
    qk_max: np.ndarray
    k_norm: np.ndarray
    v_norm: np.ndarray
    v_contrib: np.ndarray
    entropy: np.ndarray
    hidden_norm: np.ndarray
    hidden_update: np.ndarray
    hidden_rotation: np.ndarray
    top_group: np.ndarray
    #: Head-averaged attention over the *full* source axis, kept only for the
    #: handful of figure samples: ``[L, S, T]``.
    full_attention: Optional[np.ndarray] = None

    @property
    def density(self) -> np.ndarray:
        """``mass / |group|``, with empty groups left as NaN rather than zero."""
        size = self.group_size[None, None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(size > 0, self.mass / np.maximum(size, 1e-9), np.nan)
        return out


class _GroupCollector:
    """Accumulates one forward pass' statistics; driven by the patched attention."""

    def __init__(self, group_index: torch.Tensor, target_positions: torch.Tensor,
                 n_layers: int, keep_full: bool = False):
        # group_index: (T,) long, -1 for unassigned; target_positions: (S,) long
        self.group_index = group_index
        self.targets = target_positions
        self.n_layers = n_layers
        self.keep_full = keep_full
        self.layers: Dict[int, Dict[str, np.ndarray]] = {}
        self._full: Dict[int, np.ndarray] = {}

        total = group_index.numel()
        n_groups = len(GROUPS)
        onehot = torch.zeros(n_groups, total)
        valid = group_index >= 0
        onehot[group_index[valid], torch.arange(total)[valid]] = 1.0
        self.onehot = onehot                                  # (G, T)
        # A target at position p can only see j <= p, so the per-target group size
        # is not the group size: "density" over invisible tokens would be a lie.
        causal = (torch.arange(total)[None, :] <= target_positions[:, None]).float()
        self.visible = causal                                 # (S, T)
        self.group_size = (causal @ onehot.T).numpy()         # (S, G)
        self._on_device: Dict[torch.device, Tuple[torch.Tensor, ...]] = {}

    def _device_tensors(self, device):
        """Move the masks once, not once per layer: this runs 32 times per forward."""
        if device not in self._on_device:
            self._on_device[device] = (
                self.onehot.to(device=device, dtype=torch.float32),
                self.visible.to(device=device, dtype=torch.float32),
                self.targets.to(device),
                self.group_index.to(device),
                (self.visible[None, :, :] * self.onehot[:, None, :]).to(
                    device=device, dtype=torch.float32),
            )
        return self._on_device[device]

    def record(self, layer_idx: int, pre_logits: torch.Tensor,
               probs: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               scaling: float) -> None:
        """``pre_logits``/``probs``: (H, T, T); ``key``/``value``: (H, T, d)."""
        device = probs.device
        onehot, visible, targets, group_of, visible_group = self._device_tensors(device)

        p = probs.index_select(1, targets).float()                        # (H, S, T)
        logits = pre_logits.index_select(1, targets).float()              # (H, S, T)

        mass = torch.einsum("hst,gt->hsg", p, onehot)

        counts = visible_group.sum(-1)                                    # (G, S)
        qk_sum = torch.einsum("hst,gst->hsg", logits, visible_group)
        qk_mean = qk_sum / counts.T[None].clamp_min(1e-9)
        qk_mean = torch.where(counts.T[None] > 0, qk_mean,
                              torch.full_like(qk_mean, float("nan")))

        very_negative = torch.finfo(logits.dtype).min
        qk_max = torch.stack([
            logits.masked_fill(visible_group[g][None] <= 0, very_negative).amax(-1)
            for g in range(onehot.size(0))], dim=-1)                      # (H, S, G)
        qk_max = torch.where(counts.T[None] > 0, qk_max,
                             torch.full_like(qk_max, float("nan")))

        # ||sum_{j in g} A_tj v_j||: what the group actually contributed to the
        # attention output, which attention weight alone does not determine.
        contrib = torch.stack([
            torch.bmm((p * onehot[g][None, None, :]), value.float()).norm(dim=-1)
            for g in range(onehot.size(0))], dim=-1)                      # (H, S, G)

        k_norm = self._grouped_norm(key.float(), onehot)
        v_norm = self._grouped_norm(value.float(), onehot)

        entropy = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)  # (H, S)
        top_group = group_of[p.argmax(-1)]                                  # (H, S)

        self.layers[layer_idx] = {
            "mass": mass.cpu().numpy(),
            "qk_mean": qk_mean.cpu().numpy(),
            "qk_max": qk_max.cpu().numpy(),
            "k_norm": k_norm.cpu().numpy(),
            "v_norm": v_norm.cpu().numpy(),
            "v_contrib": contrib.cpu().numpy(),
            "entropy": entropy.cpu().numpy(),
            "top_group": top_group.cpu().numpy(),
        }
        if self.keep_full:
            self._full[layer_idx] = p.mean(0).to(torch.float16).cpu().numpy()

    @staticmethod
    def _grouped_norm(x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        """Mean ``||x_j||`` over each group; ``x`` is (H, T, d)."""
        norms = x.norm(dim=-1)                                            # (H, T)
        total = torch.einsum("ht,gt->hg", norms, onehot)
        counts = onehot.sum(-1)[None, :]
        out = total / counts.clamp_min(1e-9)
        return torch.where(counts > 0, out, torch.full_like(out, float("nan")))

    def finish(self, hidden_states: Optional[Sequence[torch.Tensor]]) -> CollectedAttention:
        if len(self.layers) != self.n_layers:
            raise RuntimeError(
                f"collected {len(self.layers)} layers but the model has {self.n_layers}; "
                "the attention patch did not see every layer")
        order = sorted(self.layers)

        def stack(key: str) -> np.ndarray:
            return np.stack([self.layers[i][key] for i in order], axis=0)

        shape = (self.n_layers + 1, len(GROUPS))
        hidden_norm = np.full(shape, np.nan, dtype=np.float32)
        hidden_update = np.full((self.n_layers, len(GROUPS)), np.nan, dtype=np.float32)
        hidden_rotation = np.full((self.n_layers, len(GROUPS)), np.nan, dtype=np.float32)
        if hidden_states is not None:
            onehot = self._device_tensors(hidden_states[0].device)[0]
            counts = onehot.sum(-1)

            def per_group(values: torch.Tensor) -> np.ndarray:
                totals = onehot @ values
                out = torch.where(counts > 0, totals / counts.clamp_min(1e-9),
                                  torch.full_like(totals, float("nan")))
                return out.cpu().numpy()

            previous = None
            for i, state in enumerate(hidden_states):
                current = state[0].float()                                 # (T, h)
                hidden_norm[i] = per_group(current.norm(dim=-1))
                if previous is not None:
                    delta = (current - previous).norm(dim=-1)
                    hidden_update[i - 1] = per_group(
                        delta / previous.norm(dim=-1).clamp_min(1e-9))
                    hidden_rotation[i - 1] = per_group(
                        torch.nn.functional.cosine_similarity(current, previous, dim=-1))
                previous = current

        full = None
        if self.keep_full:
            full = np.stack([self._full[i] for i in order], axis=0)

        return CollectedAttention(
            mass=stack("mass"), group_size=self.group_size.astype(np.float32),
            qk_mean=stack("qk_mean"), qk_max=stack("qk_max"),
            k_norm=stack("k_norm"), v_norm=stack("v_norm"),
            v_contrib=stack("v_contrib"), entropy=stack("entropy"),
            hidden_norm=hidden_norm, hidden_update=hidden_update,
            hidden_rotation=hidden_rotation, top_group=stack("top_group"),
            full_attention=full)


@contextmanager
def capture_attention(collector: Optional[_GroupCollector]):
    """Run the decoder with ``eager_attention_forward`` replaced by a recording copy.

    Recomputing the statistics from a returned ``[layers, heads, T, T]`` tensor is
    not an option at HotpotQA's K=10 (a raw-text row is ~1400 positions, i.e. 4 GB
    of attention weights), and ``output_attentions`` cannot expose the *pre-softmax*
    logits at all -- which is exactly the quantity that separates SeleCom's
    scale explanation from a plain token-count effect.  So the arithmetic is
    reproduced here, one layer at a time, and
    ``tests/test_attention_grouping.py`` checks it against a hand-computed
    single-step attention and against the unpatched implementation.
    """
    from transformers.models.mistral import modeling_mistral as mistral

    if collector is None:
        yield
        return

    original = mistral.eager_attention_forward

    def recording(module, query, key, value, attention_mask, scaling,
                  dropout: float = 0.0, **kwargs):
        key_states = mistral.repeat_kv(key, module.num_key_value_groups)
        value_states = mistral.repeat_kv(value, module.num_key_value_groups)

        pre = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        weights = pre
        if attention_mask is not None:
            weights = weights + attention_mask[:, :, :, : key_states.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32
                                              ).to(query.dtype)
        weights = torch.nn.functional.dropout(weights, p=dropout, training=module.training)
        output = torch.matmul(weights, value_states)
        output = output.transpose(1, 2).contiguous()

        if query.size(0) != 1:
            raise ValueError("the diagnostic forward runs one row at a time so that "
                             "padding cannot shift the group manifest")
        collector.record(int(module.layer_idx), pre[0], weights[0],
                         key_states[0], value_states[0], scaling)
        return output, weights

    mistral.eager_attention_forward = recording
    try:
        yield
    finally:
        mistral.eager_attention_forward = original


@torch.no_grad()
def diagnostic_forward(lm, inputs_embeds: torch.Tensor, group_index: torch.Tensor,
                       target_positions: torch.Tensor, keep_full: bool = False
                       ) -> CollectedAttention:
    """One teacher-forced forward that yields every mechanism statistic at once.

    Teacher forcing rather than free generation for the headline heatmaps
    (instruction §8.1): a compressed and an uncompressed run diverge after their
    first generated token, and then any attention difference is partly a
    difference in what the model already wrote.  Forcing the same target tokens
    into both removes that confound.
    """
    if inputs_embeds.size(0) != 1:
        raise ValueError("diagnostic_forward takes one row at a time")
    n_layers = len(lm.get_decoder().layers) if hasattr(lm, "get_decoder") \
        else len(lm.model.layers)
    collector = _GroupCollector(group_index, target_positions, n_layers, keep_full)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                device=inputs_embeds.device)
    with capture_attention(collector):
        output = lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    use_cache=False, output_hidden_states=True)
    return collector.finish(output.hidden_states), output


def dominance_ratios(collected: CollectedAttention, numerator: str = "document",
                     denominator: str = "instruction",
                     epsilon: float = 1e-6) -> Dict[str, float]:
    """The two pre-registered ratios from instruction §8.3, averaged over L, H, S.

    Both are reported always.  Mass alone rises with group length, so on its own it
    would "confirm" SeleCom for PISCO at K=10 purely because 80 memory slots
    outnumber a 20-token instruction; density is what asks whether an individual
    memory position is unusually attractive.
    """
    a, b = GROUP_INDEX[numerator], GROUP_INDEX[denominator]
    mass, density = collected.mass, collected.density
    size = collected.group_size
    if not np.isfinite(size[:, b]).all() or size[:, b].max() <= 0:
        return {"mass_ratio": float("nan"), "density_ratio": float("nan"),
                "qk_gap": float("nan")}
    with _quiet_nan():
        mass_ratio = np.nanmean(mass[..., a]) / (np.nanmean(mass[..., b]) + epsilon)
        density_ratio = np.nanmean(density[..., a]) / (np.nanmean(density[..., b]) + epsilon)
        qk_gap = float(np.nanmean(collected.qk_mean[..., a])
                       - np.nanmean(collected.qk_mean[..., b]))
    return {"mass_ratio": float(mass_ratio), "density_ratio": float(density_ratio),
            "qk_gap": qk_gap}


def head_dominance(collected: CollectedAttention, numerator: str = "document",
                   denominator: str = "instruction") -> Dict[str, float]:
    """Is the effect a property of the model, or of a handful of heads? (§8.5)"""
    a, b = GROUP_INDEX[numerator], GROUP_INDEX[denominator]
    with _quiet_nan():
        top = np.nanmean(collected.density[..., a], axis=2)      # (L, H)
        bottom = np.nanmean(collected.density[..., b], axis=2)
    per_head = top > bottom
    ratio = top / np.maximum(bottom, 1e-9)
    finite = ratio[np.isfinite(ratio)]
    return {
        "dominant_head_fraction": float(np.mean(per_head)),
        "head_density_ratio_median": float(np.median(finite)) if finite.size else float("nan"),
        "head_density_ratio_iqr": float(np.subtract(*np.percentile(finite, [75, 25])))
        if finite.size else float("nan"),
    }


# --------------------------------------------------------------------------------------
# 5. Behavioural metrics
# --------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def longest_common_span(prediction: str, document: str) -> int:
    """Longest run of consecutive document words reproduced verbatim.

    Token overlap alone cannot tell "answered using the document" from "recited
    the document"; a long contiguous copied span can.
    """
    p, d = _tokens(prediction), _tokens(document)
    if not p or not d:
        return 0
    previous = [0] * (len(d) + 1)
    best = 0
    for i in range(1, len(p) + 1):
        current = [0] * (len(d) + 1)
        for j in range(1, len(d) + 1):
            if p[i - 1] == d[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            current[j] = (previous[j - 1] + 1 if a[i - 1] == b[j - 1]
                          else max(previous[j], current[j - 1]))
        previous = current
    return previous[len(b)]


def rouge_l(prediction: str, reference: str) -> float:
    p, r = _tokens(prediction), _tokens(reference)
    if not p or not r:
        return float(p == r)
    lcs = _lcs_length(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    p, r = _tokens(prediction), _tokens(reference)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


_LEADING_FILLER = re.compile(
    r"^(sure|okay|ok|certainly|here(\s+is|'s)?( the)?( string)?|output|answer|response)"
    r"[\s:,\-]*", re.IGNORECASE)


def score_conflict(prediction: str, nonce: str, document: str,
                   nonce_token_ids: Optional[Sequence[int]] = None,
                   tokenizer=None) -> Dict[str, float]:
    """Did the model follow the instruction, or stay with the document anyway?

    The obvious rate -- "does the nonce appear in the output" -- is wrong, and the
    first smoke run showed exactly how: PISCO answers *"I'm sorry, but the provided
    document does not contain the string QMBW...; the document only describes the
    Liber Paradisus"*.  That quotes the nonce while refusing the instruction and
    reciting the document, i.e. it is the failure SeleCom describes, yet a
    substring test scores it as perfect compliance.  So three graded rates are
    reported and the headline one is ``leading``:

    ``exact``     the whole output is the nonce, which is what was asked;
    ``leading``   the output *starts* with the nonce, allowing "Sure: <nonce>";
    ``mentions``  the nonce appears anywhere -- an upper bound that includes
                  refusals, never an instruction-following rate.

    ``nonce_share`` is the continuous version: the fraction of output words that
    are the nonce, near 1 for a compliant answer and near 0 for a refusal that
    happens to quote it.
    """
    stripped = prediction.strip().strip('"').strip()
    without_filler = _LEADING_FILLER.sub("", stripped).strip().strip('"').strip()
    upper_nonce = nonce.upper()
    words = _tokens(prediction)
    nonce_word = nonce.lower()

    document_tokens = set(_tokens(document))
    overlap = (sum(1 for t in words if t in document_tokens) / max(1, len(words)))

    token_accuracy = 0.0
    if nonce_token_ids and tokenizer is not None:
        # Tokenised from the de-quoted output: a leading '"' would otherwise make
        # a perfectly compliant answer score zero on its very first token.
        predicted = tokenizer(without_filler, add_special_tokens=False)["input_ids"]
        matched = 0
        for a, b in zip(predicted, nonce_token_ids):
            if a != b:
                break
            matched += 1
        token_accuracy = matched / max(1, len(nonce_token_ids))

    return {
        "exact": float(without_filler.upper() == upper_nonce),
        "leading": float(without_filler.upper().startswith(upper_nonce)),
        "mentions": float(upper_nonce in prediction.upper()),
        "nonce_share": (sum(1 for w in words if w == nonce_word) / max(1, len(words))),
        "nonce_token_accuracy": token_accuracy,
        "document_overlap": overlap,
        "longest_copied_span": float(longest_common_span(prediction, document)),
        "n_output_words": float(len(words)),
        "empty": float(not prediction.strip()),
    }


def score_reconstruction(prediction: str, document: str) -> Dict[str, float]:
    return {
        "exact": float(_tokens(prediction) == _tokens(document)),
        "rouge_l": rouge_l(prediction, document),
        "token_f1": token_f1(prediction, document),
        "longest_copied_span": float(longest_common_span(prediction, document)),
        "copied_span_ratio": longest_common_span(prediction, document)
        / max(1, len(_tokens(document))),
    }


@torch.no_grad()
def target_nll(lm, inputs_embeds: torch.Tensor, target_positions: torch.Tensor,
               target_ids: torch.Tensor) -> Dict[str, float]:
    """Teacher-forced NLL of the target, and the log-probability of its first token.

    A behavioural rate is a thresholded quantity and hides a near miss; the NLL
    says whether the model was close to complying at all.
    """
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                device=inputs_embeds.device)
    logits = lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                use_cache=False).logits.float()
    # A causal LM's logit at t predicts token t+1, so the target at position p is
    # read off position p-1.
    predict_from = target_positions - 1
    selected = logits[0, predict_from]
    log_probs = torch.log_softmax(selected, dim=-1)
    chosen = log_probs[torch.arange(target_ids.numel(), device=log_probs.device),
                       target_ids]
    return {"nll": float(-chosen.mean().item()),
            "first_token_logprob": float(chosen[0].item()),
            "n_target_tokens": int(target_ids.numel())}


# --------------------------------------------------------------------------------------
# 6. Paired statistics
# --------------------------------------------------------------------------------------

def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = 2000,
                     seed: int = 0) -> Dict[str, float]:
    """CI on the paired mean difference ``a - b``, resampling *examples*.

    Resampling examples rather than (example, layer, head) triples: the layers and
    heads of one forward pass are not independent draws, and treating them as such
    would shrink every interval until nothing was ever inconclusive.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"paired bootstrap needs matched arrays, got {x.shape} and {y.shape}")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    difference = x - y
    rng = np.random.default_rng(seed)
    index = rng.integers(0, difference.size, size=(n_resamples, difference.size))
    means = difference[index].mean(axis=1)
    return {"delta": float(difference.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "n": int(difference.size)}


def spearman(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Rank correlation between a mechanism statistic and a behavioural outcome."""
    from scipy import stats

    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(keep.sum())}
    result = stats.spearmanr(x[keep], y[keep])
    return {"rho": float(result.statistic), "p": float(result.pvalue),
            "n": int(keep.sum())}
