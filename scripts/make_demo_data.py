"""
合成"能判别 query-as-Q 是否有用"的 RAG 数据。

=== 为什么要重写 ===
第一版是「一篇文档 = 一个 query = 一个全局唯一的答案」。这种设定下 document -> answer 是单射，
query 里没有任何文档本身不含的信息，模型认出是哪篇文档就能背出答案。结果就是 query-agnostic
的消融组和 query-as-Q 主实验都能到 100%，消融**没有判别力**。

=== 这一版的三个设计约束 ===
1. **一篇文档配 k 个不同的 query，答案各不相同**。P 个压缩 embedding 是 query-agnostic 时必须
   同时装下 k 个事实，是 query 引导时只需装下 1 个 —— 这是信息论层面的差异，也正是本工作的论点。
2. **同类型事实成对出现**。每篇文档取 3 个"类型"（人名 / 年份 / 计数 / 机构 / 材料），每个类型放
   2 条角色不同的事实（例如"主持修复的人"和"第一部专著的作者"都是人名）。这样光靠"答案是不是人名"
   猜不出来，query 必须区分**角色**。
3. **答案从有界池里随机采样，doc -> answer 非单射**。同一个答案会出现在不同文档里，记忆没用；
   同时答案都在词表内，避免 toy tokenizer 出 UNK。

=== 三份切分 ===
    train.jsonl        训练文档 × (k-1) 个 query
    eval_seen.jsonl    训练文档 × 留出的第 k 个 query   -> 文档见过、问题没问过：测 query 路由
    eval_unseen.jsonl  未见文档 × 全部 k 个 query       -> 测真正的泛化

用法:
    python scripts/make_demo_data.py --n_train_docs 24 --n_eval_docs 8 --out_dir data/multiq
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

# --------------------------------------------------------------------------------------
# 主题（文档谈论的对象）
# --------------------------------------------------------------------------------------
TOPICS = [
    ("the Verrin Basin", "geology"), ("the Ashgate Observatory", "astronomy"),
    ("the Kestrel Line railway", "transport"), ("the Marlowe Codex", "manuscripts"),
    ("the Tanager Reef survey", "marine biology"), ("the Halvard Foundry", "industrial history"),
    ("the Pellucid Archive", "library science"), ("the Orrin Valley wind array", "energy"),
    ("the Brookmere Aqueduct", "civil engineering"), ("the Sable Dune Reserve", "ecology"),
    ("the Corran Salt Flats", "geology"), ("the Ivell Bell Tower", "architecture"),
    ("the Norbeck Tramway", "transport"), ("the Quillon Herbarium", "botany"),
    ("the Ferrow Lighthouse", "maritime history"), ("the Aldmere Kiln Site", "archaeology"),
    ("the Vantry Canal Locks", "civil engineering"), ("the Selwick Peat Beds", "ecology"),
    ("the Padrig Print Works", "industrial history"), ("the Ossory Star Charts", "astronomy"),
    ("the Callowfen Weir", "hydrology"), ("the Brindle Glass House", "architecture"),
    ("the Naismith Seed Bank", "botany"), ("the Ryehope Colliery", "industrial history"),
    ("the Lammas Field System", "archaeology"), ("the Culvert Row Almshouses", "social history"),
    ("the Thornaby Signal Box", "transport"), ("the Wexley Tide Mill", "maritime history"),
    ("the Garrow Moor Cairns", "archaeology"), ("the Pellamy Bequest", "library science"),
    ("the Silloth Fish Weirs", "marine biology"), ("the Adderstone Quarry", "geology"),
]

# --------------------------------------------------------------------------------------
# 答案池：同类型池子共享，保证 doc -> answer 不是单射
# --------------------------------------------------------------------------------------
POOLS: Dict[str, List[str]] = {
    "person": ["Ivo Nasreddin", "Delphine Okonkwo", "Marek Halloran", "Sunniva Bracewell",
               "Teodor Ampofo", "Rhiannon Vasquez", "Casimir Odell", "Ingrid Pemberton",
               "Osric Lindqvist", "Beatriu Fontaine", "Yusuf Carrow", "Helena Draycott"],
    "year":   ["1798", "1826", "1841", "1873", "1889", "1902", "1917", "1934",
               "1948", "1961", "1975", "1988"],
    "count":  ["seventeen", "twenty three", "thirty one", "forty seven", "fifty two",
               "sixty eight", "seventy four", "eighty six", "ninety three", "one hundred nine"],
    "org":    ["the Corvidae Trust", "the Marrow Institute", "the Denholm Society",
               "the Ballater Endowment", "the Vantage Guild", "the Otterbourne Fund",
               "the Pellamy Bequest Board", "the Wrenfield Academy", "the Larkhill Consortium",
               "the Ashmount Chapter"],
    "material": ["banded serpentinite", "grey millstone grit", "cast phosphor bronze",
                 "seasoned larch timber", "glazed terracotta", "riveted wrought iron",
                 "polished travertine", "hand pressed brick", "green Westmorland slate",
                 "laminated ash board"],
}

# --------------------------------------------------------------------------------------
# 事实槽位：每个 type 两个 role，逼 query 去区分角色而不是只区分答案类型
#   (type, role, 问句模板, 陈述句模板)
# --------------------------------------------------------------------------------------
SLOTS = [
    ("person", "restorer",
     "Who directed the restoration of {topic}?",
     "The restoration of {topic} was directed by {answer}, whose site notes survive in full."),
    ("person", "author",
     "Who wrote the first monograph on {topic}?",
     "The first monograph on {topic} was written by {answer} and is still widely cited."),

    ("year", "catalogued",
     "In what year was {topic} first catalogued?",
     "{topic_cap} was first catalogued in {answer}, well before the surrounding area was mapped."),
    ("year", "closed",
     "In what year was {topic} closed to the public?",
     "{topic_cap} was closed to the public in {answer}, and has been opened only by arrangement since."),

    ("count", "chambers",
     "How many chambers were recorded at {topic}?",
     "The detailed survey recorded {answer} chambers at {topic}, a count never since revised."),
    ("count", "months",
     "How many months did construction at {topic} take?",
     "Construction at {topic} took {answer} months from first excavation to final inspection."),

    ("org", "funder",
     "Which body funded the survey of {topic}?",
     "The survey of {topic} was funded by {answer}, an arrangement unusual for the period."),
    ("org", "custodian",
     "Which body now holds the records of {topic}?",
     "The surviving records of {topic} are now held by {answer} under a single accession."),

    ("material", "original",
     "What material was used in the original construction of {topic}?",
     "The original construction of {topic} used {answer}, imported at considerable expense."),
    ("material", "facing",
     "What material was used for the later facing of {topic}?",
     "The later facing of {topic} was carried out in {answer}, chosen for its weathering."),
]

TYPES = ["person", "year", "count", "org", "material"]

FILLER = [
    "Early accounts of {topic} are fragmentary, and most surviving descriptions were compiled long afterwards.",
    "Scholars of {field} have argued that the record should be read alongside regional accounts.",
    "Funding was intermittent, and several planned phases were postponed or quietly abandoned.",
    "Local newspapers covered the work sporadically, often repeating figures that later proved unreliable.",
    "A revised inventory reorganised the material by provenance instead of by date of acquisition.",
    "Weather during the survey season was unusually poor, which delayed measurements by several weeks.",
    "Later commentators disagreed about how much of the original design survived the renovations.",
    "Administrative responsibility changed hands twice, leaving gaps in the correspondence held today.",
    "Visitors were admitted only on appointment, and the logbooks record fewer than two hundred names.",
    "A committee recommended further study but did not specify a timetable or a budget.",
    "The technical vocabulary used in the reports was inconsistent, complicating any comparison.",
    "Photographic documentation exists for only part of the site, and the negatives are unevenly preserved.",
    "Correspondence about {topic} was bound out of order, which has misled more than one commentator.",
    "Measurements were recorded in two different systems of units without any note of the conversion.",
    "The site was used for storage during the intervening decades, and little was written down.",
]


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------------------
def make_document(doc_idx: int, topic: str, field: str, rng: random.Random,
                  n_types: int, n_filler: int) -> Dict:
    """造一篇文档，返回 {document, facts:[{query, answer, gold_sentence, span, type, role}]}。"""
    chosen_types = rng.sample(TYPES, k=n_types)

    facts = []
    for t in chosen_types:
        slots = [s for s in SLOTS if s[0] == t]                 # 同一类型的两个 role 一起放进来
        answers = rng.sample(POOLS[t], k=len(slots))            # 同文档内同类型答案互不相同
        for (typ, role, q_tpl, a_tpl), ans in zip(slots, answers):
            facts.append({
                "type": typ, "role": role,
                "query": q_tpl.format(topic=topic),
                "answer": ans,
                "sentence": a_tpl.format(topic=topic, topic_cap=_cap(topic), answer=ans),
            })
    rng.shuffle(facts)

    # 段落 = 若干干扰段 + 每条事实各自一段（事实句藏在干扰句中间，位置随机）
    paragraphs: List[Dict] = []
    for _ in range(n_filler):
        sents = rng.sample(FILLER, k=3)
        paragraphs.append({"text": " ".join(s.format(topic=topic, field=field) for s in sents),
                           "fact": None})
    for f in facts:
        sents = [s.format(topic=topic, field=field) for s in rng.sample(FILLER, k=2)]
        pos = rng.randrange(len(sents) + 1)
        sents.insert(pos, f["sentence"])
        paragraphs.append({"text": " ".join(sents), "fact": f})
    rng.shuffle(paragraphs)

    document = "\n\n".join(p["text"] for p in paragraphs)

    # 定位每条事实句在全文中的字符区间
    for p_idx, p in enumerate(paragraphs):
        f = p["fact"]
        if f is None:
            continue
        start = document.index(f["sentence"])
        assert document.count(f["sentence"]) == 1, "事实句在文档里不唯一，span 会有歧义"
        f["gold_char_span"] = [start, start + len(f["sentence"])]
        f["gold_paragraph_idx"] = p_idx

    return {"doc_id": f"doc-{doc_idx:03d}", "topic": topic, "document": document, "facts": facts}


def rows_from(doc: Dict, facts: List[Dict], split: str) -> List[Dict]:
    out = []
    for j, f in enumerate(facts):
        out.append({
            "id": f"{doc['doc_id']}-{f['type']}-{f['role']}",
            "doc_id": doc["doc_id"],
            "split": split,
            "document": doc["document"],
            "query": f["query"],
            "answer": f["answer"],
            "gold_sentence": f["sentence"],
            "gold_char_span": f["gold_char_span"],
            "gold_paragraph_idx": f["gold_paragraph_idx"],
            "fact_type": f["type"],
            "fact_role": f["role"],
            "n_facts_in_doc": len(doc["facts"]),
        })
    return out


def write_jsonl(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train_docs", type=int, default=24)
    ap.add_argument("--n_eval_docs", type=int, default=8)
    ap.add_argument("--n_types", type=int, default=3, help="每篇文档取几个事实类型（每类型 2 条事实）")
    ap.add_argument("--n_filler", type=int, default=6, help="纯干扰段落数")
    ap.add_argument("--out_dir", default="data/multiq")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_docs = args.n_train_docs + args.n_eval_docs
    assert n_docs <= len(TOPICS), f"主题不够用：最多 {len(TOPICS)} 篇文档"
    assert 1 <= args.n_types <= len(TYPES)

    rng = random.Random(args.seed)
    docs = [make_document(i, *TOPICS[i], rng=rng, n_types=args.n_types, n_filler=args.n_filler)
            for i in range(n_docs)]

    train_docs, eval_docs = docs[: args.n_train_docs], docs[args.n_train_docs:]

    train_rows, seen_rows, unseen_rows = [], [], []
    for d in train_docs:                       # 每篇留出最后一条事实作为"文档见过、问题没问过"
        train_rows += rows_from(d, d["facts"][:-1], "train")
        seen_rows += rows_from(d, d["facts"][-1:], "eval_seen")
    for d in eval_docs:                        # 未见文档，全部事实都拿来测
        unseen_rows += rows_from(d, d["facts"], "eval_unseen")

    out = args.out_dir
    write_jsonl(os.path.join(out, "train.jsonl"), train_rows)
    write_jsonl(os.path.join(out, "eval_seen.jsonl"), seen_rows)
    write_jsonl(os.path.join(out, "eval_unseen.jsonl"), unseen_rows)

    k = args.n_types * 2
    meta = {
        "n_train_docs": len(train_docs), "n_eval_docs": len(eval_docs),
        "facts_per_doc": k, "queries_per_train_doc": k - 1,
        "rows": {"train": len(train_rows), "eval_seen": len(seen_rows), "eval_unseen": len(unseen_rows)},
        "doc_words_mean": round(sum(len(d["document"].split()) for d in docs) / len(docs), 1),
        "answer_pool_sizes": {t: len(v) for t, v in POOLS.items()},
        "seed": args.seed,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"-> {out}/{{train,eval_seen,eval_unseen}}.jsonl")


if __name__ == "__main__":
    main()
