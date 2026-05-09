import argparse
import json
from pathlib import Path

from src.indexing.inverted_index import InvertedIndex
from src.retrieval.boolean_retrieval import BooleanRetriever


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shorten(text, max_len=300):
    text = " ".join(str(text).split())
    return text[:max_len] + "..." if len(text) > max_len else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["boolean"], default="boolean")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    corpus_path = Path("data/processed/corpus.json")
    index_path = Path("outputs/indexes/inverted_index.json")

    if not corpus_path.exists():
        raise FileNotFoundError("Chưa có corpus.json. Hãy chạy prepare_data.py trước.")

    if not index_path.exists():
        raise FileNotFoundError("Chưa có inverted_index.json. Hãy chạy inverted_index.py trước.")

    corpus = load_json(corpus_path)
    index = InvertedIndex.load(index_path)

    retriever = BooleanRetriever(index)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Method: {args.method}")
    print()

    for rank, item in enumerate(results, start=1):
        doc_id = item["doc_id"]
        print(f"{rank}. CID: {doc_id}")
        print(f"   Score: {item['score']}")
        print(f"   Text: {shorten(corpus.get(doc_id, ''))}")
        print()


if __name__ == "__main__":
    main()