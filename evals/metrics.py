# -*- coding: utf-8 -*-
"""Метрики качества RAG: hit@k, MRR, полнота ответа (must_contain).

Принцип: метрики считаются по golden set до того, как что-то меняешь
в промпте/поиске/конвейере. Порог в CI (см. run_evals.py) - это контракт.
"""


def hit_at_k(runs, k=3):
    """Доля вопросов, где нужный док оказался в топ-k выдачи."""
    if not runs:
        return 0.0
    hits = sum(
        1 for r in runs if r["expected_doc"] in r["retrieved_ids"][:k]
    )
    return hits / len(runs)


def mrr(runs):
    """Mean Reciprocal Rank."""
    if not runs:
        return 0.0
    total = 0.0
    for r in runs:
        rank = None
        for i, doc_id in enumerate(r["retrieved_ids"]):
            if doc_id == r["expected_doc"]:
                rank = i + 1
                break
        if rank:
            total += 1.0 / rank
    return total / len(runs)


def answer_coverage(runs):
    """Доля обязательных фрагментов (must_contain), попавших в ответ."""
    if not runs:
        return 0.0
    total, found = 0, 0
    for r in runs:
        for frag in r["must_contain"]:
            total += 1
            if frag.lower() in r["answer"].lower():
                found += 1
    return found / max(1, total)


def citation_rate(runs):
    """Доля ответов, где цитируется ровно тот документ, что ожидался."""
    if not runs:
        return 0.0
    ok = sum(
        1
        for r in runs
        if r["citations"] and r["citations"][0]["id"] == r["expected_doc"]
    )
    return ok / len(runs)
