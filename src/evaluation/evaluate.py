import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation.metrics import evaluate_query
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.tfidf_retrieval import TfidfRetriever
from src.utils.console import configure_utf8_stdout


EVALUATION_K = 10
DEFAULT_MAX_QUERIES = 100
METHODS = ("boolean", "tfidf", "bm25")
REPORT_PATH = Path("outputs/reports/evaluation_report.csv")
INDEX_PATH = Path("outputs/indexes/inverted_index.pkl")
TFIDF_MODEL_PATH = Path("outputs/indexes/tfidf_model.pkl")
BM25_MODEL_PATH = Path("outputs/indexes/bm25_model.pkl")


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_top_k(value: str) -> int:
    top_k = int(value.strip())
    if top_k <= 0:
        raise ValueError("--top_k phai la so nguyen duong.")
    return top_k


def build_retriever(method: str, corpus: dict[str, str]):
    expected_num_docs = len(corpus)

    if method == "boolean":
        if INDEX_PATH.exists():
            index = InvertedIndex.load_pickle(INDEX_PATH)
            if index.num_docs == expected_num_docs:
                return BooleanRetriever(index)
        index = InvertedIndex()
        index.build(corpus)
        return BooleanRetriever(index)

    if method == "tfidf":
        if TFIDF_MODEL_PATH.exists():
            retriever = TfidfRetriever.load(TFIDF_MODEL_PATH)
            if retriever.num_docs == expected_num_docs:
                return retriever
        retriever = TfidfRetriever()
        retriever.fit(corpus)
        return retriever

    if method == "bm25":
        if BM25_MODEL_PATH.exists():
            retriever = BM25Retriever.load(BM25_MODEL_PATH)
            if retriever.num_docs == expected_num_docs:
                return retriever
        retriever = BM25Retriever()
        retriever.fit(corpus)
        return retriever

    raise ValueError(f"Unsupported method: {method}")


def normalize_max_queries(max_queries: int | None) -> int | None:
    if max_queries is None or max_queries <= 0:
        return None
    return max_queries


def average_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}

    keys = metric_rows[0].keys()
    return {
        key: sum(row[key] for row in metric_rows) / len(metric_rows)
        for key in keys
    }


def evaluate_method(
    method: str,
    corpus: dict[str, str],
    queries: dict[str, str],
    qrels: dict[str, list[str]],
    top_k: int,
    max_queries: int | None = None,
) -> dict[str, float]:
    retriever = build_retriever(method, corpus)
    query_items = list(queries.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    per_query_metrics = []
    for qid, query_text in tqdm(query_items, desc=f"Evaluating {method}"):
        results = retriever.search(query_text, top_k=top_k)
        retrieved_doc_ids = [str(item["doc_id"]) for item in results]
        relevant_doc_ids = {str(doc_id) for doc_id in qrels.get(qid, [])}
        per_query_metrics.append(
            evaluate_query(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=relevant_doc_ids,
                k=EVALUATION_K,
            )
        )

    averages = average_metrics(per_query_metrics)
    averages["method"] = method
    averages["num_queries"] = len(query_items)
    averages["top_k"] = str(top_k)
    return averages


def save_report(rows: list[dict[str, float]], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "num_queries",
        "top_k",
        "precision@10",
        "recall@10",
        "map",
        "ndcg@10",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0.0) for field in fieldnames})


def print_report(rows: list[dict[str, float]]) -> None:
    headers = [
        "method",
        "num_queries",
        "top_k",
        "precision@10",
        "recall@10",
        "map",
        "ndcg@10",
    ]
    print(" | ".join(header.ljust(14) for header in headers))
    print("-" * 108)
    for row in rows:
        values = [
            str(row.get("method", "")).ljust(14),
            str(row.get("num_queries", "")).ljust(14),
            str(row.get("top_k", "")).ljust(14),
            f"{row.get('precision@10', 0.0):.4f}".ljust(14),
            f"{row.get('recall@10', 0.0):.4f}".ljust(14),
            f"{row.get('map', 0.0):.4f}".ljust(14),
            f"{row.get('ndcg@10', 0.0):.4f}".ljust(14),
        ]
        print(" | ".join(values))


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(description="Evaluate retrieval methods")
    parser.add_argument("--method", choices=["boolean", "tfidf", "bm25", "all"], default="all")
    parser.add_argument("--top_k", default="10")
    parser.add_argument("--max_queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--corpus_path", default="data/processed/corpus.json")
    parser.add_argument("--queries_path", default="data/processed/queries.json")
    parser.add_argument("--qrels_path", default="data/processed/qrels.json")
    args = parser.parse_args()

    top_k = parse_top_k(args.top_k)
    methods = list(METHODS) if args.method == "all" else [args.method]
    corpus = load_json(args.corpus_path)
    queries = load_json(args.queries_path)
    qrels = load_json(args.qrels_path)
    max_queries = normalize_max_queries(args.max_queries)

    if max_queries is None:
        print("Dang evaluation tren toan bo query.")
    else:
        print(f"Dang evaluation tren {max_queries} query dau tien. Dung --max_queries 0 de chay full.")

    print(f"Che do retrieval: top_k={top_k}.")

    report_rows = []

    for method in methods:
        report_rows.append(
            evaluate_method(
                method=method,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                top_k=top_k,
                max_queries=max_queries,
            )
        )

    save_report(report_rows, REPORT_PATH)
    print_report(report_rows)
    print(f"\nDa luu bao cao tai: {REPORT_PATH}")


if __name__ == "__main__":
    main()
