"""Answer-distribution distillation from the full-cache teacher.

The student reads 8 soft tokens; P reads all 80 latents of the same cache and
scores 54.50 EM against the student's 43.35.  The gap is *defined* relative to P,
so P is the natural teacher: same documents, same frozen Mistral, same tokenizer,
same answer tokens.  What distillation transfers is the teacher's distribution
over each answer token, which carries more per example than the one-hot gold --
not "dense gradient where there was none", since cross-entropy already has dense
gradients w.r.t. the logits.

Two things this cannot do, and the writeup must not claim:

* **It cannot exceed the teacher.**  The student's input is strictly a
  compression of the teacher's, so parity is the ceiling.  Parity at a tenth of
  the evidence tokens is the goal; the method is 11.7 points short of it.
* **It cannot fix a wrong teacher.**  P is wrong on plenty of questions, which is
  why the gold cross-entropy stays in the loss rather than being replaced.

**Top-k caching is not a renormalised distribution.**  Storing the teacher's top-k
logits and softmaxing over those k gives a different distribution from the
teacher's -- it silently deletes the tail and rescales everything else.  The cache
therefore stores probabilities normalised over the *full* vocabulary plus the
leftover tail mass, and the divergence is computed with the tail as one aggregate
bucket:

    KL ~= sum_{v in topk} pT(v) log[pT(v)/pS(v)]  +  pT(tail) log[pT(tail)/pS(tail)]

That is an aggregate approximation of the true KL, not the KL itself; it is exact
only when the teacher's tail is a single token.  It lower-bounds the true value,
because merging the tail into one bucket cannot increase divergence.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Guards the logs.  Probabilities below this are indistinguishable from zero in
#: fp16 storage anyway, so clamping costs no fidelity.
EPS = 1e-9


def teacher_probabilities(logits: torch.Tensor, top_k: int, temperature: float):
    """Full-vocabulary softmax at ``temperature``, kept as top-k plus tail mass.

    Softmax first, truncate second.  The reverse -- truncate then softmax -- is
    the mistake this module exists to avoid.
    """
    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    top = probabilities.topk(min(top_k, probabilities.size(-1)), dim=-1)
    tail = (1.0 - top.values.sum(-1)).clamp_min(0.0)
    return top.indices, top.values, tail


def distillation_loss(student_logits: torch.Tensor,
                      teacher_index: torch.Tensor,
                      teacher_probability: torch.Tensor,
                      teacher_tail: torch.Tensor,
                      temperature: float = 2.0) -> torch.Tensor:
    """Aggregated KL(teacher || student) over answer positions.

    ``student_logits``      (N, V) at the positions predicting answer tokens
    ``teacher_index``       (N, k) vocabulary ids the teacher put its mass on
    ``teacher_probability`` (N, k) full-vocabulary probabilities at ``temperature``
    ``teacher_tail``        (N,)   the mass outside the top-k

    Computed in fp32 and averaged over positions.  The caller multiplies by
    ``temperature ** 2``: softening the distributions scales the gradients by
    ``1/T**2``, so without it the effective weight on this term changes whenever
    the temperature does, and a temperature sweep silently becomes a weight sweep.
    """
    if student_logits.ndim != 2:
        raise ValueError(f"student_logits must be (N, V), got {tuple(student_logits.shape)}")
    if teacher_index.shape != teacher_probability.shape:
        raise ValueError("teacher index and probability must have the same shape")
    if teacher_index.size(0) != student_logits.size(0):
        raise ValueError(
            f"{teacher_index.size(0)} teacher rows against "
            f"{student_logits.size(0)} student rows; the answer-relative "
            "alignment is broken")

    log_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    log_top = log_student.gather(-1, teacher_index)                      # (N, k)
    # The student's tail is whatever is left after the teacher's top-k, so the two
    # distributions are compared over the same partition of the vocabulary.
    top_mass = log_top.exp().sum(-1).clamp(max=1.0 - EPS)
    log_tail = torch.log1p(-top_mass)

    probability = teacher_probability.float()
    tail = teacher_tail.float().clamp_min(0.0)
    divergence = (probability * (probability.clamp_min(EPS).log() - log_top)).sum(-1)
    divergence = divergence + tail * (tail.clamp_min(EPS).log() - log_tail)
    # Teacher mass can fall slightly short of 1 after fp16 round-trip; the sum is
    # then not a probability distribution and the divergence can go marginally
    # negative.  Clamp rather than let a negative loss term reward the student.
    return divergence.clamp_min(0.0).mean()


class TeacherCache:
    """Precomputed teacher distributions, addressed by example id.

    Only training examples are ever stored.  Distilling on dev or test would make
    the teacher a channel from the evaluation data into the student.
    """

    def __init__(self, payload: dict):
        self.meta = payload["meta"]
        self.index = payload["index"]                # id -> (offset, length)
        self.topk_index = payload["topk_index"]      # (total, k) int32
        self.topk_probability = payload["topk_probability"]
        self.tail = payload["tail"]
        self.targets = payload["targets"]            # (total,) the answer token ids

    @classmethod
    def load(cls, path: str):
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    @property
    def temperature(self) -> float:
        return float(self.meta["temperature"])

    def gather(self, ids, orders, device):
        """Teacher rows for ``(example id, answer-relative index)`` pairs.

        Returns the three tensors plus a boolean mask: an example the teacher
        never saw, or an answer longer than the teacher's, contributes nothing
        rather than being matched to the wrong position.
        """
        rows, keep = [], []
        for example, order in zip(ids, orders):
            location = self.index.get(example)
            if location is None or order >= location[1]:
                rows.append(0)
                keep.append(False)
            else:
                rows.append(location[0] + order)
                keep.append(True)
        rows = torch.as_tensor(rows, dtype=torch.long)
        keep = torch.as_tensor(keep, dtype=torch.bool)
        return (self.topk_index[rows].to(device).long(),
                self.topk_probability[rows].to(device),
                self.tail[rows].to(device),
                keep.to(device),
                self.targets[rows].to(device))
