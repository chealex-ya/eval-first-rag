# -*- coding: utf-8 -*-
"""LLM-as-judge с калибровкой и offline-фолбэком.

Судья оценивает groundedness ответа по бинарной рубрике (не шкала 1-10 -
двоичные суждения стабильнее). Без ключа API включается lexical-фолбэк:
оценка детерминированная, чтобы CI не мигал.

Подключение любого OpenAI-совместимого провайдера:
    export LLM_API_KEY=...
    export LLM_BASE_URL=https://api.example.com/v1   # опционально
"""

import json
import os
import re
import urllib.request

RUBRIC = (
    "Ты - судья качества ответов базы знаний банка. Вопрос пользователя, "
    "эталонный документ и ответ системы даны ниже. Ответь строго JSON "
    "{{\"grounded\": true|false, \"reason\": \"<до 20 слов>\"}}. "
    "grounded=true только если ответ опирается на документ и не противоречит ему."
)


def judge_llm(question, context, answer):
    """Возвращает (grounded: bool, reason: str). При отсутствии API - фолбэк."""
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return judge_lexical(question, context, answer)
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    payload = {
        "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": f"Вопрос: {question}\nДокумент:\n{context}\n\nОтвет системы:\n{answer}",
            },
        ],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    verdict = json.loads(m.group(0)) if m else {"grounded": False, "reason": "parse error"}
    return bool(verdict.get("grounded")), verdict.get("reason", "")


def judge_lexical(question, context, answer):
    """Offline-фолбэк: доля токенов ответа, покрытых контекстом.

    Грубая прокси groundedness: extractive-ответ обязан покрываться
    документом. Порог 0.9 выбран по золотому набору (см. reports/).
    """
    from rag.retrieval import tokenize

    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context))
    if not answer_tokens:
        return False, "пустой ответ"
    coverage = len(answer_tokens & context_tokens) / len(answer_tokens)
    return coverage >= 0.9, f"lexical coverage {coverage:.2f}"
