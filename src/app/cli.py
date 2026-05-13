import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.tfidf_retrieval import TfidfRetriever
from src.utils.console import configure_utf8_stdout


CORPUS_PATH = Path("data/processed/corpus.json")
INDEX_PATH = Path("outputs/indexes/inverted_index.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def shorten(text: str, max_len: int = 300) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len].rstrip() + "..."


def get_retriever(method: str, corpus: dict[str, str]):
    if method == "boolean":
        if INDEX_PATH.exists():
            index = InvertedIndex.load(INDEX_PATH)
        else:
            index = InvertedIndex()
            index.build(corpus)
            index.save(INDEX_PATH)
        return BooleanRetriever(index)

    if method == "tfidf":
        retriever = TfidfRetriever()
        retriever.fit(corpus)
        return retriever

    if method == "bm25":
        retriever = BM25Retriever()
        retriever.fit(corpus)
        return retriever

    raise ValueError(f"Unsupported method: {method}")


def main():
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Vietnamese Legal IR CLI")
    parser.add_argument("--method", choices=["boolean", "tfidf", "bm25"], default="bm25")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Chua co data processed. Hay chay python -m src.data.prepare_data truoc."
        )

    corpus = load_json(CORPUS_PATH)
    retriever = get_retriever(args.method, corpus)
    results = retriever.search(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Method: {args.method}")
    print()

    if not results:
        print("Khong tim thay ket qua phu hop.")
        return

    print(f"Top {args.top_k} results:")
    print()
    for rank, item in enumerate(results, start=1):
        doc_id = item["doc_id"]
        print(f"{rank}. CID: {doc_id}")
        print(f"   Score: {item['score']:.6f}")
        print(f"   Text: {shorten(corpus.get(doc_id, ''))}")
        print()


if __name__ == "__main__":
    main()
