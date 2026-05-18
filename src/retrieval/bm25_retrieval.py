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


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = 0
        self.doc_ids = []
        self.avgdl = 0.0
        self.document_frequency = {}
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.postings = defaultdict(dict)
        self.idf = {}

    def fit(self, corpus: dict[str, str]) -> None:
        self.num_docs = len(corpus)
        self.doc_ids = [str(doc_id) for doc_id in corpus.keys()]
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.postings = defaultdict(dict)
        document_frequency = Counter()

        for doc_id, text in tqdm(corpus.items(), desc="Fitting BM25"):
            doc_id = str(doc_id)
            tokens = preprocess(text)
            term_counts = Counter(tokens)

            self.doc_lengths[doc_id] = len(tokens)
            self.doc_term_freqs[doc_id] = dict(term_counts)

            for term, freq in term_counts.items():
                self.postings[term][doc_id] = freq
            document_frequency.update(term_counts.keys())

        total_length = sum(self.doc_lengths.values())
        self.avgdl = total_length / self.num_docs if self.num_docs else 0.0
        self.document_frequency = dict(document_frequency)
        self.idf = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self.document_frequency.items()
        }

    @staticmethod
    def _finalize_results(
        scored_results: list[dict],
        top_k: int,
    ) -> list[dict]:
        if not scored_results:
            return []

        if top_k <= 0:
            return []

        top_results = heapq.nlargest(
            min(top_k, len(scored_results)),
            scored_results,
            key=lambda item: (item["score"], item["doc_id"]),
        )
        top_results.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return top_results

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        query_terms = preprocess(query)
        if not query_terms:
            return []

        unique_query_terms = [term for term in dict.fromkeys(query_terms) if term in self.postings]
        if not unique_query_terms:
            return []

        scores = defaultdict(float)
        for term in unique_query_terms:
            postings = self.postings.get(term, {})
            term_idf = self.idf.get(term, 0.0)
            for doc_id, tf in postings.items():
                dl = self.doc_lengths.get(doc_id, 0)
                if dl == 0 or self.avgdl == 0:
                    continue

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += term_idf * (numerator / denominator)

        scored_results = [
            {
                "doc_id": doc_id,
                "score": float(score),
                "method": "bm25",
            }
            for doc_id, score in scores.items()
            if score > 0
        ]
        return self._finalize_results(scored_results, top_k=top_k)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "k1": self.k1,
            "b": self.b,
            "num_docs": self.num_docs,
            "doc_ids": self.doc_ids,
            "avgdl": self.avgdl,
            "document_frequency": self.document_frequency,
            "doc_lengths": self.doc_lengths,
            "doc_term_freqs": self.doc_term_freqs,
            "postings": {term: doc_map for term, doc_map in self.postings.items()},
            "idf": self.idf,
        }

        if output_path.suffix.lower() == ".json":
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
            return

        with output_path.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        input_path = Path(path)
        if input_path.suffix.lower() == ".json":
            with input_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        else:
            with input_path.open("rb") as file:
                data = pickle.load(file)

        retriever = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        retriever.num_docs = data.get("num_docs", 0)
        retriever.doc_ids = data.get("doc_ids", [])
        retriever.avgdl = data.get("avgdl", 0.0)
        retriever.document_frequency = data.get("document_frequency", {})
        retriever.doc_lengths = data.get("doc_lengths", {})
        retriever.doc_term_freqs = data.get("doc_term_freqs", {})
        retriever.postings = defaultdict(dict, data.get("postings", {}))
        retriever.idf = data.get("idf", {})
        return retriever


def load_corpus(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="BM25 retrieval CLI test")
    parser.add_argument("--corpus_path", default="data/processed/corpus.json")
    parser.add_argument("--query", default="dieu kien ket hon")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    retriever = BM25Retriever(k1=args.k1, b=args.b)
    retriever.fit(corpus)

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Top {args.top_k} ket qua:")
    for rank, item in enumerate(results, start=1):
        print(f"{rank}. doc_id={item['doc_id']} score={item['score']:.6f} method={item['method']}")


if __name__ == "__main__":
    main()
