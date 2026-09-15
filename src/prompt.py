"""Decoder-side prompt assembly, kept token-identical to PISCO's.

QuRO's generator *is* PISCO's decoder, so the prompt must be the one that decoder
was trained on (``third_party/modelling_pisco.py:1036``).  Anything else throws
away the warm start and makes the main table incomparable with the PISCO
baseline.  The only change is the number of memory slots: PISCO emits ``k * m``
of them, QuRO emits its readout budget ``B``.

``DecoderInputMode`` covers the "does the decoder still need the question in
plain text?" question.  In every prior soft-compression system the answer is
trivially yes: the compressed tokens are query-agnostic, so matching evidence
against the question is work the decoder has to do.  QuRO moves that matching
into the readout, which makes ``D1`` -- same prompt, question text removed --
both a meaningful ablation and a usable training regime: with no plain-text
query, the only path for query information is through the soft tokens, so the
loss cannot go down unless the readout genuinely conditions on the query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

SYSTEM_PROMPT = (
    "You are a helpful assistant. Your task is to extract relevant information "
    "from provided documents and to answer to questions as briefly as possible."
)

#: D0 pisco-identical | D1 no question text | D2 question first | D3 slots only
DECODER_INPUT_MODES = ("D0", "D1", "D2", "D3")


@dataclass
class BuiltPrompt:
    input_ids: List[int]
    slot_positions: List[int]


class PiscoPromptBuilder:
    """Render the PISCO chat prompt with exactly ``budget`` memory slots."""

    def __init__(self, tokenizer, n_mem_tokens: int, mode: str = "D0",
                 system_prompt: str = SYSTEM_PROMPT):
        if mode not in DECODER_INPUT_MODES:
            raise ValueError(f"decoder_input_mode must be one of {DECODER_INPUT_MODES}, got {mode}")
        self.tok = tokenizer
        self.mode = mode
        self.system_prompt = system_prompt
        self.n_mem_tokens = int(n_mem_tokens)
        self.mem_tokens: Sequence[str] = tokenizer.mem_tokens
        self.mem_token_ids = set(tokenizer.mem_token_ids)
        self.sep_token = getattr(tokenizer, "sep_token", "")

    def slot_string(self, budget: int) -> str:
        """``budget`` slots laid out as PISCO's blocks of ``n_mem_tokens`` + ``<SEP>``.

        Keeping the block structure means QuRO at ``B = k * m`` produces a token
        sequence identical to PISCO at top-k, so the two differ only in the
        embeddings written into those slots.
        """
        if budget < 1:
            raise ValueError("budget must be >= 1")
        out, remaining = [], budget
        while remaining > 0:
            take = min(self.n_mem_tokens, remaining)
            out.append("".join(self.mem_tokens[:take]) + self.sep_token)
            remaining -= take
        return "".join(out)

    def _render(self, query: str, budget: int) -> str:
        slots = self.slot_string(budget)
        if self.mode == "D3":
            return slots
        if self.mode == "D0":
            user = f"Background:\n{slots}\n\nQuestion:{query}"
        elif self.mode == "D1":
            # Format anchor kept, every query token removed: the delta against D0
            # is exactly the plain-text question and nothing else.
            user = f"Background:\n{slots}\n\nQuestion:"
        else:  # D2
            user = f"Question:{query}\n\nBackground:\n{slots}"

        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user.replace(":\\ ", ": ")}]
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Mistral-v0.2 has no system role; PISCO folds it into the user turn.
            merged = [{"role": "user", "content": self.system_prompt + "\n" + user}]
            return self.tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def build(self, query: str, budget: int) -> BuiltPrompt:
        text = self._render(query, budget)
        input_ids = self.tok(text, add_special_tokens=False, truncation=True,
                             max_length=2048)["input_ids"]
        if self.mode == "D3":
            bos = getattr(self.tok, "bos_token_id", None)
            input_ids = ([bos] if bos is not None else []) + input_ids
        positions = [i for i, token in enumerate(input_ids) if token in self.mem_token_ids]
        if len(positions) != budget:
            raise ValueError(
                f"prompt has {len(positions)} memory slots but budget is {budget}; "
                "the prompt was probably truncated")
        return BuiltPrompt(input_ids, positions)


def assemble_inputs(
    generator_embeddings,
    prompts: Sequence[BuiltPrompt],
    soft_tokens: torch.Tensor,
    soft_token_mask: torch.Tensor,
    target_ids: Optional[Sequence[Sequence[int]]] = None,
    pad_token_id: int = 0,
    pad_side: str = "right",
):
    """Embed prompts, write soft tokens into the memory slots, append targets.

    ``pad_side="right"`` for training (labels mask the padding) and ``"left"`` for
    batched generation (``generate`` requires the real tokens to end flush right).
    """
    device = soft_tokens.device
    dtype = generator_embeddings.weight.dtype
    hidden = generator_embeddings.weight.size(1)
    sequences, labels = [], []

    for i, prompt in enumerate(prompts):
        ids = torch.tensor(prompt.input_ids, device=device)
        embeds = generator_embeddings(ids).clone()
        valid = soft_tokens[i][soft_token_mask[i]].to(dtype)
        if valid.size(0) != len(prompt.slot_positions):
            raise ValueError(
                f"row {i}: {valid.size(0)} soft tokens for {len(prompt.slot_positions)} slots")
        embeds[torch.tensor(prompt.slot_positions, device=device)] = valid
        label = [-100] * embeds.size(0)
        if target_ids is not None and len(target_ids[i]):
            target = torch.tensor(list(target_ids[i]), device=device)
            embeds = torch.cat([embeds, generator_embeddings(target)])
            label = label + list(target_ids[i])
        sequences.append(embeds)
        labels.append(label)

    width = max(x.size(0) for x in sequences)
    inputs = generator_embeddings(
        torch.full((len(sequences), width), pad_token_id, device=device)).clone()
    attention = torch.zeros(len(sequences), width, dtype=torch.long, device=device)
    packed_labels = torch.full((len(sequences), width), -100, dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        n = sequence.size(0)
        start = width - n if pad_side == "left" else 0
        inputs[i, start:start + n] = sequence
        attention[i, start:start + n] = 1
        packed_labels[i, start:start + n] = torch.tensor(labels[i], device=device)

    out = {"inputs_embeds": inputs.to(dtype), "attention_mask": attention}
    if target_ids is not None:
        out["labels"] = packed_labels
    return out
