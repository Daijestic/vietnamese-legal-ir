import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.data.load_dataset import DATASET_NAME, load_legal_dataset


PROCESSED_DIR = Path("data/processed")
CORPUS_PATH = PROCESSED_DIR / "corpus.json"
QUERIES_PATH = PROCESSED_DIR / "queries.json"
QRELS_PATH = PROCESSED_DIR / "qrels.json"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def normalize_id_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def normalize_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def find_corpus_split(dataset) -> str | None:
    for split_name in dataset.keys():
        split = dataset[split_name]
        columns = set(split.column_names)
        if {"cid", "text"}.issubset(columns):
            return split_name
    return None


def build_corpus_from_split(dataset, split_name: str, max_docs: int | None) -> dict[str, str]:
    corpus = {}
    split = dataset[split_name]

    for row in tqdm(split, desc=f"Processing corpus split '{split_name}'"):
        doc_id = row.get("cid")
        text = row.get("text", "")

        if doc_id is None:
            continue

        corpus[str(doc_id)] = str(text)

        if max_docs is not None and len(corpus) >= max_docs:
            break

    return corpus


def build_queries_and_qrels(
    dataset,
    existing_corpus: dict[str, str] | None = None,
    max_queries: int | None = None,
    max_docs: int | None = None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    queries = {}
    qrels = {}
    corpus = dict(existing_corpus or {})
    corpus_is_fixed = existing_corpus is not None
    total_queries = 0

    for split_name in dataset.keys():
        split = dataset[split_name]
        columns = set(split.column_names)

        if not {"qid", "question"}.issubset(columns):
            continue

        for row in tqdm(split, desc=f"Processing query split '{split_name}'"):
            if max_queries is not None and total_queries >= max_queries:
                return queries, qrels, corpus

            qid = row.get("qid")
            question = row.get("question", "")

            if qid is None:
                continue

            relevant_doc_ids = normalize_id_list(row.get("cid"))
            contexts = normalize_text_list(row.get("context_list"))

            if not corpus_is_fixed:
                for doc_id, context in zip(relevant_doc_ids, contexts):
                    if max_docs is not None and len(corpus) >= max_docs and doc_id not in corpus:
                        continue
                    corpus.setdefault(doc_id, context)

            filtered_relevant_doc_ids = [
                doc_id for doc_id in relevant_doc_ids if doc_id in corpus
            ]

            if not filtered_relevant_doc_ids:
                continue

            queries[str(qid)] = str(question)
            qrels[str(qid)] = filtered_relevant_doc_ids
            total_queries += 1

    return queries, qrels, corpus


def prepare_data(max_queries: int | None = None, max_docs: int | None = None) -> None:
    print(f"Dang load dataset: {DATASET_NAME}")
    dataset = load_legal_dataset(DATASET_NAME)

    corpus_split_name = find_corpus_split(dataset)
    corpus = {}

    if corpus_split_name is not None:
        print(f"Tim thay corpus split: {corpus_split_name}")
        corpus = build_corpus_from_split(dataset, corpus_split_name, max_docs=max_docs)
    else:
        print("Khong tim thay corpus split. Se trich corpus tu context_list trong train/test.")

    queries, qrels, corpus = build_queries_and_qrels(
        dataset,
        existing_corpus=corpus if corpus else None,
        max_queries=max_queries,
        max_docs=max_docs,
    )

    save_json(corpus, CORPUS_PATH)
    save_json(queries, QUERIES_PATH)
    save_json(qrels, QRELS_PATH)

    print("\nHoan tat chuan hoa du lieu.")
    print(f"So document trong corpus: {len(corpus)}")
    print(f"So query: {len(queries)}")
    print(f"So qrels: {len(qrels)}")
    print(f"Da luu: {CORPUS_PATH}")
    print(f"Da luu: {QUERIES_PATH}")
    print(f"Da luu: {QRELS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Chuan hoa dataset thanh corpus/queries/qrels")
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    prepare_data(max_queries=args.max_queries, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
