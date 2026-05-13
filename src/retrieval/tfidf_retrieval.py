import argparse
import heapq
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from src.preprocessing.tokenizer import preprocess
from src.utils.console import configure_utf8_stdout


class TfidfRetriever:
    def __init__(self):
        self.num_docs = 0
        self.doc_ids = []
        self.vocabulary = set()
        self.document_frequency = {}
        self.idf = {}
        self.doc_vectors = {}
        self.doc_norms = {}
        self.postings = defaultdict(dict)

    def fit(self, corpus: dict[str, str]) -> None:
        self.num_docs = len(corpus)
        self.doc_ids = [str(doc_id) for doc_id in corpus.keys()]
        doc_term_freqs = {}
        document_frequency = Counter()
        self.postings = defaultdict(dict)

        for doc_id, text in tqdm(corpus.items(), desc="Fitting TF-IDF"):
            doc_id = str(doc_id)
            tokens = preprocess(text)
            if not tokens:
                doc_term_freqs[doc_id] = {}
                continue

            term_counts = Counter(tokens)
            doc_length = len(tokens)
            tf = {
                term: count / doc_length
                for term, count in term_counts.items()
            }
            doc_term_freqs[doc_id] = tf

            document_frequency.update(term_counts.keys())

        self.document_frequency = dict(document_frequency)
        self.vocabulary = set(self.document_frequency.keys())
        self.idf = {
            term: math.log((self.num_docs + 1) / (df + 1)) + 1.0
            for term, df in self.document_frequency.items()
        }

        self.doc_vectors = {}
        self.doc_norms = {}

        for doc_id, tf_weights in doc_term_freqs.items():
            vector = {
                term: tf_value * self.idf[term]
                for term, tf_value in tf_weights.items()
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = norm
            for term, weight in vector.items():
                self.postings[term][doc_id] = weight

    def _vectorize_query(self, query: str) -> dict[str, float]:
        tokens = preprocess(query)
        if not tokens:
            return {}

        counts = Counter(token for token in tokens if token in self.idf)
        if not counts:
            return {}

        query_length = sum(counts.values())
        return {
            term: (count / query_length) * self.idf[term]
            for term, count in counts.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vector = self._vectorize_query(query)
        if not query_vector:
            return []

        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        if query_norm == 0:
            return []

        dot_products = defaultdict(float)
        for term, query_weight in query_vector.items():
            for doc_id, doc_weight in self.postings.get(term, {}).items():
                dot_products[doc_id] += query_weight * doc_weight

        if not dot_products:
            return []

        scored_results = []
        for doc_id, dot_product in dot_products.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            if doc_norm == 0:
                continue

            score = dot_product / (query_norm * doc_norm)
            if score > 0:
                scored_results.append(
                    {
                        "doc_id": doc_id,
                        "score": float(score),
                        "method": "tfidf",
                    }
                )

        if not scored_results:
            return []

        top_results = heapq.nlargest(
            min(top_k, len(scored_results)),
            scored_results,
            key=lambda item: item["score"],
        )
        top_results.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return top_results

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "num_docs": self.num_docs,
            "doc_ids": self.doc_ids,
            "vocabulary": sorted(self.vocabulary),
            "document_frequency": self.document_frequency,
            "idf": self.idf,
            "doc_vectors": self.doc_vectors,
            "doc_norms": self.doc_norms,
            "postings": {
                term: doc_map
                for term, doc_map in self.postings.items()
            },
        }

        if output_path.suffix.lower() == ".json":
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            return

        with output_path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "TfidfRetriever":
        input_path = Path(path)
        if input_path.suffix.lower() == ".json":
            with input_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            with input_path.open("rb") as file:
                data = pickle.load(file)

        retriever = cls()
        retriever.num_docs = data.get("num_docs", 0)
        retriever.doc_ids = data.get("doc_ids", [])
        retriever.vocabulary = set(data.get("vocabulary", []))
        retriever.document_frequency = data.get("document_frequency", {})
        retriever.idf = data.get("idf", {})
        retriever.doc_vectors = data.get("doc_vectors", {})
        retriever.doc_norms = data.get("doc_norms", {})
        retriever.postings = defaultdict(dict, data.get("postings", {}))
        return retriever


def load_corpus(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="TF-IDF retrieval CLI test")
    parser.add_argument("--corpus_path", default="data/processed/corpus.json")
    parser.add_argument("--query", default="điều kiện kết hôn")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    retriever = TfidfRetriever()
    retriever.fit(corpus)

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Top {args.top_k} ket qua:")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. doc_id={item['doc_id']} score={item['score']:.6f} method={item['method']}")


if __name__ == "__main__":
    main()
