# -*- coding: utf-8 -*-
"""Гибридный поиск: BM25 + TF-IDF (косинус), слияние через Reciprocal Rank Fusion.

Только stdlib - чтобы демо запускалось везде и CI был детерминированным.
В production TF-IDF-ветку заменяет dense-эмбеддинги, BM25-ветку -
SPLADE/лексический движок; RRF-слияние и интерфейс не меняются.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же "
    "вы за бы по только ее мне было вот от меня еще нет о из ему теперь "
    "когда даже ну вдруг ли если уже или ни быть был него до вас нибудь "
    "опять уж вам сказал ведь там потом себя ничего ей может они тут где "
    "есть надо ней для мы тебя их чем была сам чтоб без будто человек "
    "чего раз тоже себе под жизнь будет ж тогда кто этот говорил того "
    "потому этого какой совсем ним здесь этом один почти мой тем чтобы "
    "нее кажется сейчас были куда зачем сказать всех никогда сегодня "
    "можно при наконец два об другой хоть после над больше тот через эти "
    "нас про всего них какая много разве三 сказала три эту моя впрочем "
    "хорошо свою этой перед иногда лучше чуть том нельзя такой им более "
    "всегда конечно всю между the a an of to in is are for on with".split()
)


def tokenize(text):
    """Нормализация: нижний регистр, унификация ё->е, стоп-слова."""
    text = text.lower().replace("ё", "е")
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


class BM25:
    """Классический Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs_tokens):
        self.k1 = 1.5
        self.b = 0.75
        self.doc_count = len(docs_tokens)
        self.doc_lens = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, self.doc_count)
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {
            term: math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def score(self, query_tokens, index):
        s = 0.0
        doc_tf = self.tf[index]
        dl = self.doc_lens[index]
        for term in query_tokens:
            if term not in doc_tf or term not in self.idf:
                continue
            f = doc_tf[term]
            idf = self.idf[term]
            s += idf * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
        return s


class Tfidf:
    """TF-IDF вектора + косинусное сходство (_dense-ветка в демо)."""

    def __init__(self, docs_tokens):
        self.doc_count = len(docs_tokens)
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {t: math.log((self.doc_count + 1) / (n + 1)) + 1 for t, n in df.items()}
        self.vecs = [self._vec(d) for d in docs_tokens]

    def _vec(self, tokens):
        tf = Counter(tokens)
        # Незнакомые термины (нет в корпусе) получают сглаженный idf,
        # а не KeyError: запрос имеет право содержать новые слова.
        unseen = math.log(self.doc_count + 1) + 1
        return {t: (1 + math.log(c)) * self.idf.get(t, unseen) for t, c in tf.items()}

    def score(self, query_tokens, index):
        q = self._vec(query_tokens)
        d = self.vecs[index]
        if not q or not d:
            return 0.0
        dot = sum(w * d.get(t, 0.0) for t, w in q.items())
        nq = math.sqrt(sum(w * w for w in q.values()))
        nd = math.sqrt(sum(w * w for w in d.values()))
        return dot / (nq * nd) if nq and nd else 0.0


class HybridRetriever:
    """Слияние рангов двух веток через Reciprocal Rank Fusion (k=60)."""

    def __init__(self, documents):
        self.documents = documents
        tokens = [tokenize(d["text"]) for d in documents]
        self.bm25 = BM25(tokens)
        self.tfidf = Tfidf(tokens)
        self._cache = {}

    def _ranked(self, query, scorer):
        q = tokenize(query)
        scores = [scorer(q, i) for i in range(len(self.documents))]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return order

    def search(self, query, top_k=3):
        lex_rank = self._ranked(query, self.bm25.score)
        vec_rank = self._ranked(query, self.tfidf.score)

        rrf = {}
        for rank_list in (lex_rank, vec_rank):
            for rank, doc_index in enumerate(rank_list):
                rrf[doc_index] = rrf.get(doc_index, 0.0) + 1.0 / (60 + rank + 1)

        fused = sorted(rra_items(rrf), key=lambda kv: -kv[1])[:top_k]
        return [
            {"index": i, "doc": self.documents[i], "rrf": score}
            for i, score in fused
        ]


def rra_items(d):
    return list(d.items())
