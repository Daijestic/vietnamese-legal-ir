import argparse
import json
import time
from pathlib import Path

from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.tfidf_retrieval import TfidfRetriever
from src.utils.console import configure_utf8_stdout


ROOT_DIR = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus.json"
QUERIES_PATH = ROOT_DIR / "data" / "processed" / "queries.json"
QRELS_PATH = ROOT_DIR / "data" / "processed" / "qrels.json"
INDEX_JSON_PATH = ROOT_DIR / "outputs" / "indexes" / "inverted_index.json"
INDEX_PKL_PATH = ROOT_DIR / "outputs" / "indexes" / "inverted_index.pkl"
TFIDF_MODEL_PATH = ROOT_DIR / "outputs" / "indexes" / "tfidf_model.pkl"
BM25_MODEL_PATH = ROOT_DIR / "outputs" / "indexes" / "bm25_model.pkl"


def load_corpus(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_processed_data() -> None:
    missing_paths = [path for path in (CORPUS_PATH, QUERIES_PATH, QRELS_PATH) if not path.exists()]
    if not missing_paths:
        print("[SKIP] corpus.json already exists")
        print("[SKIP] queries.json already exists")
        print("[SKIP] qrels.json already exists")
        return

    missing_display = ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing_paths)
    raise FileNotFoundError(
        f"Missing processed data files: {missing_display}\n"
        "Hay chay: python -m src.data.prepare_data"
    )


def build_or_load_index(corpus: dict[str, str], force: bool) -> InvertedIndex:
    expected_num_docs = len(corpus)

    if force:
        print("[BUILD] rebuilding inverted index from corpus...")
        index = InvertedIndex()
        index.build(corpus)
        index.save(INDEX_JSON_PATH)
        index.save_pickle(INDEX_PKL_PATH)
        print(f"[DONE] saved {INDEX_JSON_PATH.relative_to(ROOT_DIR)}")
        print(f"[DONE] saved {INDEX_PKL_PATH.relative_to(ROOT_DIR)}")
        return index

    if INDEX_PKL_PATH.exists():
        index = InvertedIndex.load_pickle(INDEX_PKL_PATH)
        if index.num_docs == expected_num_docs:
            print("[SKIP] inverted_index.pkl already exists")
            return index
        print("[BUILD] inverted_index.pkl is out of date. Rebuilding...")
        index = InvertedIndex()
        index.build(corpus)
        index.save(INDEX_JSON_PATH)
        index.save_pickle(INDEX_PKL_PATH)
        print(f"[DONE] saved {INDEX_JSON_PATH.relative_to(ROOT_DIR)}")
        print(f"[DONE] saved {INDEX_PKL_PATH.relative_to(ROOT_DIR)}")
        return index

    if INDEX_JSON_PATH.exists():
        index = InvertedIndex.load(INDEX_JSON_PATH)
        if index.num_docs != expected_num_docs:
            print("[BUILD] inverted_index.json is out of date. Rebuilding...")
            index = InvertedIndex()
            index.build(corpus)
            index.save(INDEX_JSON_PATH)
            index.save_pickle(INDEX_PKL_PATH)
            print(f"[DONE] saved {INDEX_JSON_PATH.relative_to(ROOT_DIR)}")
            print(f"[DONE] saved {INDEX_PKL_PATH.relative_to(ROOT_DIR)}")
            return index

        print("[BUILD] converting inverted_index.json -> inverted_index.pkl ...")
        index.save_pickle(INDEX_PKL_PATH)
        print(f"[DONE] saved {INDEX_PKL_PATH.relative_to(ROOT_DIR)}")
        return index

    print("[BUILD] building inverted index...")
    index = InvertedIndex()
    index.build(corpus)
    index.save(INDEX_JSON_PATH)
    index.save_pickle(INDEX_PKL_PATH)
    print(f"[DONE] saved {INDEX_JSON_PATH.relative_to(ROOT_DIR)}")
    print(f"[DONE] saved {INDEX_PKL_PATH.relative_to(ROOT_DIR)}")
    return index


def build_or_skip_tfidf(corpus: dict[str, str], force: bool) -> None:
    if TFIDF_MODEL_PATH.exists() and not force:
        print("[SKIP] tfidf_model.pkl already exists")
        return

    print("[BUILD] fitting TF-IDF...")
    retriever = TfidfRetriever()
    retriever.fit(corpus)
    retriever.save(TFIDF_MODEL_PATH)
    print(f"[DONE] saved {TFIDF_MODEL_PATH.relative_to(ROOT_DIR)}")


def build_or_skip_bm25(corpus: dict[str, str], force: bool) -> None:
    if BM25_MODEL_PATH.exists() and not force:
        print("[SKIP] bm25_model.pkl already exists")
        return

    print("[BUILD] fitting BM25...")
    retriever = BM25Retriever()
    retriever.fit(corpus)
    retriever.save(BM25_MODEL_PATH)
    print(f"[DONE] saved {BM25_MODEL_PATH.relative_to(ROOT_DIR)}")


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Build retrieval artifacts for full-data demo")
    parser.add_argument("--force", action="store_true", help="Force rebuild index and retrieval models")
    args = parser.parse_args()

    total_start = time.perf_counter()
    ensure_processed_data()

    load_corpus_start = time.perf_counter()
    corpus = load_corpus(CORPUS_PATH)
    load_corpus_time = time.perf_counter() - load_corpus_start
    print(f"[INFO] corpus size: {len(corpus)} documents")
    print(f"[TIME] load corpus: {load_corpus_time:.2f}s")

    index_start = time.perf_counter()
    build_or_load_index(corpus, force=args.force)
    index_time = time.perf_counter() - index_start
    print(f"[TIME] load/build index: {index_time:.2f}s")

    tfidf_start = time.perf_counter()
    build_or_skip_tfidf(corpus, force=args.force)
    tfidf_time = time.perf_counter() - tfidf_start
    print(f"[TIME] fit/load TF-IDF: {tfidf_time:.2f}s")

    bm25_start = time.perf_counter()
    build_or_skip_bm25(corpus, force=args.force)
    bm25_time = time.perf_counter() - bm25_start
    print(f"[TIME] fit/load BM25: {bm25_time:.2f}s")

    total_time = time.perf_counter() - total_start
    print(f"[TIME] total: {total_time:.2f}s")


if __name__ == "__main__":
    main()
