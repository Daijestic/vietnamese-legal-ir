import argparse
from pathlib import Path
from collections import Counter
import re

from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.tokenizer import preprocess


class BooleanRetriever:
    def __init__(self, index: InvertedIndex):
        self.index = index
        self.all_doc_ids = set(index.doc_lengths.keys())

    def _postings_set(self, token: str) -> set[str]:
        return set(self.index.postings.get(token, []))

    def _and_search(self, tokens: list[str]) -> set[str]:
        if not tokens:
            return set()

        result = self._postings_set(tokens[0])

        for token in tokens[1:]:
            result = result.intersection(self._postings_set(token))

        return result

    def _or_search(self, tokens: list[str]) -> set[str]:
        result = set()

        for token in tokens:
            result = result.union(self._postings_set(token))

        return result

    def _not_search(self, positive_tokens: list[str], negative_tokens: list[str]) -> set[str]:
        positive_result = self._and_search(positive_tokens) if positive_tokens else set(self.all_doc_ids)

        negative_result = set()
        for token in negative_tokens:
            negative_result = negative_result.union(self._postings_set(token))

        return positive_result.difference(negative_result)

    def _score_docs(self, doc_ids: set[str], query_tokens: list[str]) -> list[dict]:
        scores = Counter()

        for token in query_tokens:
            for doc_id in self.index.postings.get(token, []):
                if doc_id in doc_ids:
                    scores[doc_id] += 1

        results = []

        for doc_id in doc_ids:
            results.append({
                "doc_id": doc_id,
                "score": float(scores.get(doc_id, 0)),
                "method": "boolean",
            })

        results.sort(key=lambda x: (-x["score"], x["doc_id"]))
        return results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Hỗ trợ:
        - Query tự nhiên: mặc định AND các token.
        - OR không phân biệt hoa thường.
        - NOT không phân biệt hoa thường.
        """

        raw_query = query.strip()

        if not raw_query:
            return []

        if re.search(r"\bNOT\b", raw_query, flags=re.IGNORECASE):
            left, right = re.split(
                r"\bNOT\b",
                raw_query,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            positive_tokens = preprocess(left)
            negative_tokens = preprocess(right)

            if not positive_tokens:
                return []

            matched_docs = self._not_search(positive_tokens, negative_tokens)
            query_tokens = positive_tokens

        elif re.search(r"\bOR\b", raw_query, flags=re.IGNORECASE):
            parts = re.split(
                r"\bOR\b",
                raw_query,
                flags=re.IGNORECASE,
            )

            query_tokens = []

            for part in parts:
                query_tokens.extend(preprocess(part))

            if not query_tokens:
                return []

            matched_docs = self._or_search(query_tokens)

        else:
            query_tokens = preprocess(raw_query)

            if not query_tokens:
                return []

            matched_docs = self._and_search(query_tokens)

        ranked_results = self._score_docs(matched_docs, query_tokens)
        return ranked_results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Boolean Retrieval CLI test")
    parser.add_argument(
        "--index_path",
        default="outputs/indexes/inverted_index.json",
        help="Đường dẫn inverted index",
    )
    parser.add_argument(
        "--query",
        default="điều kiện kết hôn",
        help="Câu truy vấn",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Số kết quả trả về",
    )

    args = parser.parse_args()

    index_path = Path(args.index_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {index_path}. Hãy chạy: python src/indexing/inverted_index.py"
        )

    index = InvertedIndex.load(index_path)
    retriever = BooleanRetriever(index)

    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Top {args.top_k} kết quả:")

    for rank, item in enumerate(results, start=1):
        print(f"{rank}. doc_id={item['doc_id']} score={item['score']} method={item['method']}")


if __name__ == "__main__":
    main()