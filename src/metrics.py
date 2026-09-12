"""
开放域 QA 的标准指标。答案是**别名列表**，命中任一即算对。

三个指标各有各的用处，一起看才不会被单一数字骗：
    substring : 归一化后 gold 是否是预测的子串。软压缩的生成常常带前后缀，
                这个指标最宽松，也是 RECOMP / xRAG 这类工作报的 "accuracy"。
    em        : 完全相等。最严，但对"答案 + 多余解释"这种输出会全判错。
    f1        : token 级 F1。介于两者之间，能反映"答对了一半"。

归一化沿用 SQuAD 的做法：小写、去冠词、去标点、压空白。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(pred: str, golds: Sequence[str]) -> Dict[str, float]:
    """对每个别名算一遍，取最好的那个。"""
    golds = [g for g in golds if str(g).strip()] or [""]
    np_ = normalize_answer(pred)
    return {
        "substring": float(any(normalize_answer(g) in np_ for g in golds)),
        "em": float(any(normalize_answer(g) == np_ for g in golds)),
        "f1": max(_f1(pred, g) for g in golds),
    }


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"substring": 0.0, "em": 0.0, "f1": 0.0, "n": 0}
    keys = ("substring", "em", "f1")
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["n"] = len(rows)
    return out
