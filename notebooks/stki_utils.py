"""
stki_utils.py
Kumpulan fungsi hasil Praktikum Sistem Temu Kembali Informasi (STKI).

Setiap fungsi di sini dibangun langkah demi langkah di notebook pertemuan
yang bersangkutan. Notebook pertemuan berikutnya mengimpor fungsi dari file
ini agar tidak perlu menulis ulang kode dari pertemuan sebelumnya.

  Pertemuan 1 : preprocessing  (tokenize, case_folding, remove_stopwords, stem, preprocess)
  Pertemuan 2 : indexing       (build_inverted_index, build_positional_index)
  Pertemuan 3 : Boolean        (boolean_search)
  Pertemuan 4 : VSM / TF-IDF   (TfidfModel)
  Pertemuan 5 : BM25           (BM25Model)
  Pertemuan 6 : evaluasi       (precision_at_k, recall_at_k, average_precision, ndcg_at_k, ...)
  Pertemuan 8 : feedback       (BM25Model.search_weighted)
"""
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_corpus():
    """Mengembalikan DataFrame korpus (doc_id, judul, isi)."""
    return pd.read_csv(DATA_DIR / "korpus.csv")


def load_queries():
    """Mengembalikan dict {query_id: teks_query}."""
    df = pd.read_csv(DATA_DIR / "queries.csv")
    return dict(zip(df.query_id, df["query"]))


def load_qrels():
    """Mengembalikan dict {query_id: {doc_id: nilai_relevansi}}."""
    df = pd.read_csv(DATA_DIR / "qrels.csv")
    qrels = defaultdict(dict)
    for q, d, r in df.itertuples(index=False):
        qrels[q][d] = int(r)
    return dict(qrels)


def corpus_texts(df=None):
    """Mengembalikan dict {doc_id: judul + isi}."""
    df = load_corpus() if df is None else df
    return {r.doc_id: f"{r.judul}. {r.isi}" for r in df.itertuples(index=False)}


# ---------------------------------------------------------------------------
# Pertemuan 1: Preprocessing
# ---------------------------------------------------------------------------
_stemmer = StemmerFactory().create_stemmer()
STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
_stem_cache = {}


def case_folding(text):
    return text.lower()


def tokenize(text):
    """Memecah teks menjadi token berupa huruf/angka (tanda baca dibuang)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def remove_stopwords(tokens, stopwords=STOPWORDS):
    return [t for t in tokens if t not in stopwords]


def stem(token):
    if token not in _stem_cache:
        _stem_cache[token] = _stemmer.stem(token)
    return _stem_cache[token]


def preprocess(text, use_stopword=True, use_stemming=True):
    tokens = tokenize(case_folding(text))
    if use_stopword:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = [stem(t) for t in tokens]
    return tokens


# ---------------------------------------------------------------------------
# Pertemuan 2: Inverted index
# ---------------------------------------------------------------------------
def build_inverted_index(docs_tokens):
    """docs_tokens: {doc_id: [token, ...]} -> {term: {doc_id: tf}}"""
    index = defaultdict(dict)
    for doc_id, tokens in docs_tokens.items():
        for term, tf in Counter(tokens).items():
            index[term][doc_id] = tf
    return dict(sorted(index.items()))


def build_positional_index(docs_tokens):
    """-> {term: {doc_id: [posisi, ...]}}"""
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, tokens in docs_tokens.items():
        for pos, term in enumerate(tokens):
            index[term][doc_id].append(pos)
    return {t: dict(p) for t, p in sorted(index.items())}


# ---------------------------------------------------------------------------
# Pertemuan 3: Boolean retrieval
# ---------------------------------------------------------------------------
def boolean_search(query, index, all_docs):
    """
    Query Boolean sederhana dengan operator AND, OR, NOT (huruf kapital) dan
    tanda kurung, contoh: "(padi OR sawah) AND NOT hama".
    Prioritas: NOT > AND > OR. Istilah dipreprocess (stemming) seperti dokumen.
    """
    all_docs = set(all_docs)
    tokens = re.findall(r"\(|\)|[^\s()]+", query)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def parse_or():
        result = parse_and()
        while peek() == "OR":
            take()
            result = result | parse_and()
        return result

    def parse_and():
        result = parse_not()
        while peek() == "AND":
            take()
            result = result & parse_not()
        return result

    def parse_not():
        if peek() == "NOT":
            take()
            return all_docs - parse_not()
        if peek() == "(":
            take()
            result = parse_or()
            if take() != ")":
                raise ValueError("Kurung tidak seimbang")
            return result
        term = preprocess(take(), use_stopword=False)
        return set(index.get(term[0], {})) if term else set()

    result = parse_or()
    if pos != len(tokens):
        raise ValueError(f"Query tidak valid di dekat '{tokens[pos]}'")
    return sorted(result)


# ---------------------------------------------------------------------------
# Pertemuan 4: Vector Space Model (TF-IDF + cosine)
# ---------------------------------------------------------------------------
class TfidfModel:
    """TF log (1 + log10 tf), IDF log10(N/df), normalisasi panjang vektor (cosine)."""

    def __init__(self, docs_tokens):
        self.index = build_inverted_index(docs_tokens)
        self.N = len(docs_tokens)
        self.idf = {t: math.log10(self.N / len(p)) for t, p in self.index.items()}
        self.doc_vectors = {}
        for doc_id, tokens in docs_tokens.items():
            vec = {t: (1 + math.log10(tf)) * self.idf[t] for t, tf in Counter(tokens).items()}
            self.doc_vectors[doc_id] = _normalize(vec)

    def query_vector(self, q_tokens):
        vec = {t: (1 + math.log10(tf)) * self.idf[t]
               for t, tf in Counter(q_tokens).items() if t in self.idf}
        return _normalize(vec)

    def search(self, q_tokens, k=10):
        qv = self.query_vector(q_tokens)
        scores = defaultdict(float)
        for t, wq in qv.items():
            for doc_id in self.index[t]:
                scores[doc_id] += wq * self.doc_vectors[doc_id][t]
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


def _normalize(vec):
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return {t: w / norm for t, w in vec.items()} if norm > 0 else {}


# ---------------------------------------------------------------------------
# Pertemuan 5: BM25 (Okapi)
# ---------------------------------------------------------------------------
class BM25Model:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.index = build_inverted_index(docs_tokens)
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = {d: len(t) for d, t in docs_tokens.items()}
        self.avgdl = sum(self.doc_len.values()) / self.N
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.index.items()}

    def score_term(self, term, doc_id):
        tf = self.index.get(term, {}).get(doc_id, 0)
        if tf == 0:
            return 0.0
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl)
        return self.idf[term] * tf * (self.k1 + 1) / (tf + norm)

    def search(self, q_tokens, k=10):
        scores = defaultdict(float)
        for t in q_tokens:
            for doc_id in self.index.get(t, {}):
                scores[doc_id] += self.score_term(t, doc_id)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

    def search_weighted(self, q_weights, k=10):
        """Seperti search, tetapi setiap term query punya bobot: {term: bobot} (Pertemuan 8)."""
        scores = defaultdict(float)
        for t, w in q_weights.items():
            for doc_id in self.index.get(t, {}):
                scores[doc_id] += w * self.score_term(t, doc_id)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]


# ---------------------------------------------------------------------------
# Pertemuan 6: Evaluasi
# ---------------------------------------------------------------------------
def precision_at_k(ranked, relevant, k):
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked, relevant, k):
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant) if relevant else 0.0


def f1_score(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision(ranked, relevant):
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant) if relevant else 0.0


def dcg_at_k(gains, k):
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(ranked, graded, k):
    """graded: {doc_id: nilai_relevansi}"""
    gains = [graded.get(d, 0) for d in ranked[:k]]
    ideal = sorted(graded.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / idcg if idcg > 0 else 0.0


def evaluate(run, qrels, k=5):
    """run: {query_id: [doc_id terurut]} -> DataFrame metrik per query + baris rata-rata."""
    rows = []
    for q, ranked in run.items():
        graded = qrels.get(q, {})
        rel = set(graded)
        p, r = precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k)
        rows.append({"query_id": q, f"P@{k}": p, f"R@{k}": r, f"F1@{k}": f1_score(p, r),
                     "AP": average_precision(ranked, rel), f"nDCG@{k}": ndcg_at_k(ranked, graded, k)})
    df = pd.DataFrame(rows).set_index("query_id")
    df.loc["Rata-rata"] = df.mean()
    return df.round(4)
