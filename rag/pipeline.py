# -*- coding: utf-8 -*-
"""RAG-конвейер: retrieve -> compose. Ответ собирается только из найденных
чанков и несёт цитаты - источник ответа всегда проверяем.

LLM-генерация опциональна (см. evals/judge.py): без API конвейер работает
в extractive-режиме, с API - полный abstractive-режим.
"""

from rag.retrieval import HybridRetriever
from rag.corpus import CORPUS


class RagPipeline:
    def __init__(self, corpus=None):
        self.retriever = HybridRetriever(corpus or CORPUS)

    def retrieve(self, question, top_k=3):
        return self.retriever.search(question, top_k=top_k)

    def answer(self, question, top_k=3):
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return {"answer": "Ничего не найдено в базе знаний.", "citations": []}
        parts = []
        citations = []
        for hit in hits:
            doc = hit["doc"]
            citations.append({"id": doc["id"], "title": doc["title"]})
            parts.append(doc["text"])
        return {
            "answer": " ".join(parts),
            "citations": citations,
            "retrieved_ids": [h["doc"]["id"] for h in hits],
        }
