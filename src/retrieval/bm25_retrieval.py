import argparse
import json
import math
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
        self.avgdl = 0.0
        self.document_frequency = {}
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.postings = defaultdict(dict)
        self.idf = {}

    def fit(self, corpus: dict[str, str]) -> None:
        self.num_docs = len(corpus)
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

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        query_terms = preprocess(query)
        if not query_terms:
            return []

        candidate_docs = set()
        unique_query_terms = [term for term in dict.fromkeys(query_terms) if term in self.postings]
        for term in unique_query_terms:
            candidate_docs.update(self.postings[term].keys())

        scored_results = []
        for doc_id in candidate_docs:
            dl = self.doc_lengths.get(doc_id, 0)
            if dl == 0 or self.avgdl == 0:
                continue

            score = 0.0
            term_freqs = self.doc_term_freqs.get(doc_id, {})

            for term in unique_query_terms:
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf.get(term, 0.0) * (numerator / denominator)

            if score > 0:
                scored_results.append(
                    {
                        "doc_id": doc_id,
                        "score": float(score),
                        "method": "bm25",
                    }
                )

        scored_results.sort(key=lambda item: (-item["score"], item["doc_id"]))
        return scored_results[:top_k]


def load_corpus(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="BM25 retrieval CLI test")
    parser.add_argument("--corpus_path", default="data/processed/corpus.json")
    parser.add_argument("--query", default="điều kiện kết hôn")
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
