# -*- coding: utf-8 -*-
"""Прогон eval-сьюта: golden set -> конвейер -> метрики -> отчет + exit-код.

Использование:
    python evals/run_evals.py                # офлайн (детерминированно)
    LLM_API_KEY=... python evals/run_evals.py  # + LLM-as-judge

Exit-код 1, если любая метрика ниже порога - это и есть eval-гейт для CI.
Запуск из корня репозитория: python evals/run_evals.py
"""

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RagPipeline
from evals.metrics import hit_at_k, mrr, answer_coverage, citation_rate
from evals.judge import judge_llm

# Пороги - контракт качества. Снижение порога = осознанное решение, а не деградация.
THRESHOLDS = {
    "hit@3": 0.90,
    "mrr": 0.85,
    "answer_coverage": 0.95,
    "citation_rate": 0.85,
    "groundedness": 0.80,
}


def load_golden(path):
    items = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    golden = load_golden(os.path.join(root, "evals", "golden.jsonl"))
    pipeline = RagPipeline()

    runs = []
    started = time.time()
    for item in golden:
        result = pipeline.answer(item["question"], top_k=3)
        context = " ".join(
            h["doc"]["text"] for h in pipeline.retrieve(item["question"], top_k=3)
        )
        grounded, reason = judge_llm(item["question"], context, result["answer"])
        runs.append(
            {
                **item,
                "answer": result["answer"],
                "citations": result["citations"],
                "retrieved_ids": result["retrieved_ids"],
                "grounded": grounded,
                "judge_reason": reason,
            }
        )
    elapsed = time.time() - started

    scores = {
        "hit@3": hit_at_k(runs, k=3),
        "mrr": mrr(runs),
        "answer_coverage": answer_coverage(runs),
        "citation_rate": citation_rate(runs),
        "groundedness": sum(1 for r in runs if r["grounded"]) / len(runs),
    }

    failed = {m: v for m, v in scores.items() if v < THRESHOLDS[m]}

    report = {
        "metrics": scores,
        "thresholds": THRESHOLDS,
        "failed": failed,
        "n_questions": len(runs),
        "elapsed_sec": round(elapsed, 2),
        "runs": [
            {
                "qid": r["qid"],
                "retrieved_ids": r["retrieved_ids"],
                "expected_doc": r["expected_doc"],
                "grounded": r["grounded"],
                "judge_reason": r["judge_reason"],
            }
            for r in runs
        ],
    }

    out_json = os.path.join(root, "reports", "eval_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with io.open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(u"Метрики (n=%d, %.1fs):" % (len(runs), elapsed))
    for m, v in scores.items():
        status = "OK " if v >= THRESHOLDS[m] else "FAIL"
        print(u"  %s %s: %.2f (порог %.2f)" % (status, m, v, THRESHOLDS[m]))

    if failed:
        print(u"\nEval-гейт НЕ пройден: %s" % ", ".join(failed))
        sys.exit(1)
    print(u"\nEval-гейт пройден.")


if __name__ == "__main__":
    main()
