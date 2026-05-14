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
INDEX_PATH = Path("outputs/indexes/inverted_index.pkl")
TFIDF_MODEL_PATH = Path("outputs/indexes/tfidf_model.pkl")
BM25_MODEL_PATH = Path("outputs/indexes/bm25_model.pkl")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def shorten(text: str, max_len: int = 300) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len].rstrip() + "..."


def parse_top_k(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized == "none":
        return None

    top_k = int(normalized)
    if top_k < 0:
        raise ValueError("--top_k phai la so nguyen khong am hoac 'none'.")
    return top_k


def get_retriever(method: str, corpus: dict[str, str]):
    expected_num_docs = len(corpus)

    if method == "boolean":
        if INDEX_PATH.exists():
            index = InvertedIndex.load_pickle(INDEX_PATH)
            if index.num_docs == expected_num_docs:
                return BooleanRetriever(index)
        index = InvertedIndex()
        index.build(corpus)
        index.save_pickle(INDEX_PATH)
        return BooleanRetriever(index)

    if method == "tfidf":
        if TFIDF_MODEL_PATH.exists():
            retriever = TfidfRetriever.load(TFIDF_MODEL_PATH)
            if retriever.num_docs == expected_num_docs:
                return retriever
        retriever = TfidfRetriever()
        retriever.fit(corpus)
        retriever.save(TFIDF_MODEL_PATH)
        return retriever

    if method == "bm25":
        if BM25_MODEL_PATH.exists():
            retriever = BM25Retriever.load(BM25_MODEL_PATH)
            if retriever.num_docs == expected_num_docs:
                return retriever
        retriever = BM25Retriever()
        retriever.fit(corpus)
        retriever.save(BM25_MODEL_PATH)
        return retriever

    raise ValueError(f"Unsupported method: {method}")


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Vietnamese Legal IR CLI")
    parser.add_argument("--method", choices=["boolean", "tfidf", "bm25"], default="bm25")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", default="5")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--display_limit", type=int, default=20)
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "Chua co data processed. Hay chay python -m src.data.prepare_data truoc."
        )

    top_k = parse_top_k(args.top_k)
    corpus = load_json(CORPUS_PATH)
    retriever = get_retriever(args.method, corpus)
    results = retriever.search(args.query, top_k=top_k, threshold=args.threshold)

    print(f"Query: {args.query}")
    print(f"Method: {args.method}")
    print(f"Threshold: {args.threshold}")
    print()

    if not results:
        print("Khong tim thay ket qua phu hop.")
        return

    total_results = len(results)
    results_to_show = results
    if top_k is None:
        display_limit = max(args.display_limit, 0)
        results_to_show = results[:display_limit]
        print(f"Retrieved {total_results} documents co score > {args.threshold}.")
        print(f"Dang hien thi {len(results_to_show)} ket qua dau tien.")
    else:
        print(f"Top {top_k} results:")

    print()
    for rank, item in enumerate(results_to_show, start=1):
        doc_id = item["doc_id"]
        print(f"{rank}. CID: {doc_id}")
        print(f"   Score: {item['score']:.6f}")
        print(f"   Text: {shorten(corpus.get(doc_id, ''))}")
        print()


if __name__ == "__main__":
    main()
