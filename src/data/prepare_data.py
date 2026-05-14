import argparse
import csv
import io
import json
import sys
try:
    csv.field_size_limit(2147483647)
except OverflowError:
    csv.field_size_limit(100000000)
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.load_dataset import load_legal_dataset
from src.utils.console import configure_utf8_stdout


DEFAULT_RAW_DATASET_NAME = "tmnam20/BKAI-Legal-Retrieval"
DEFAULT_EVAL_DATASET_NAME = "YuITC/Vietnamese-Legal-Documents"

PROCESSED_DIR = Path("data/processed")
CORPUS_PATH = PROCESSED_DIR / "corpus.json"
QUERIES_PATH = PROCESSED_DIR / "queries.json"
QRELS_PATH = PROCESSED_DIR / "qrels.json"


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def dataset_schema(dataset) -> dict[str, list[str]]:
    return {
        split_name: list(dataset[split_name].column_names)
        for split_name in dataset.keys()
    }


def normalize_id_list(value) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            normalized.extend(normalize_id_list(item))
        return normalized

    return [str(value)]


def normalize_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduplicated = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def find_split_with_columns(dataset, required_columns: set[str]) -> str:
    for split_name in dataset.keys():
        split_columns = set(dataset[split_name].column_names)
        if required_columns.issubset(split_columns):
            return split_name
    raise ValueError(
        f"Khong tim thay split chua cac cot {sorted(required_columns)} "
        f"trong dataset. Schema hien tai: {dataset_schema(dataset)}"
    )


def inspect_raw_archive_schema(raw_dataset_name: str) -> dict[str, list[str]]:
    archive_path = resolve_raw_archive_path(raw_dataset_name)
    schema = {}
    with ZipFile(archive_path) as archive:
        for member_name in archive.namelist():
            if not member_name.lower().endswith(".csv"):
                continue
            with archive.open(member_name) as file_obj:
                reader = csv.reader(io.TextIOWrapper(file_obj, encoding="utf-8-sig"))
                headers = next(reader, [])
            schema[member_name] = headers
    return schema


def resolve_raw_archive_path(raw_dataset_name: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download

        archive_path = hf_hub_download(
            repo_id=raw_dataset_name,
            repo_type="dataset",
            filename="archive.zip",
        )
        return Path(archive_path)
    except Exception as exc:
        raise RuntimeError(
            f"Khong the tai hoac tim thay archive.zip cho dataset raw '{raw_dataset_name}'."
        ) from exc


def build_corpus_from_raw_dataset(dataset, max_docs: int | None) -> tuple[dict[str, str], dict[str, str | int]]:
    split_name = find_split_with_columns(dataset, {"cid", "text"})
    split = dataset[split_name]
    corpus = {}

    for row in tqdm(split, desc=f"Processing raw corpus split '{split_name}'"):
        doc_id = row.get("cid")
        if doc_id is None:
            continue

        corpus[str(doc_id)] = normalize_text(row.get("text"))

        if max_docs is not None and len(corpus) >= max_docs:
            break

    return corpus, {
        "split_name": split_name,
        "num_docs": len(corpus),
    }


def build_corpus_from_raw_archive(
    raw_dataset_name: str,
    max_docs: int | None,
) -> tuple[dict[str, str], dict[str, str | int], dict[str, list[str]]]:
    archive_path = resolve_raw_archive_path(raw_dataset_name)
    raw_schema = inspect_raw_archive_schema(raw_dataset_name)

    if "corpus.csv" not in raw_schema:
        raise ValueError(
            f"Khong tim thay corpus.csv trong raw dataset '{raw_dataset_name}'. "
            f"Cac file CSV hien co: {sorted(raw_schema)}"
        )

    corpus = {}
    with ZipFile(archive_path) as archive:
        with archive.open("corpus.csv") as file_obj:
            reader = csv.DictReader(io.TextIOWrapper(file_obj, encoding="utf-8-sig"))
            if not {"cid", "text"}.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"corpus.csv khong chua du cac cot cid/text. Schema: {reader.fieldnames}"
                )

            for row in tqdm(reader, desc="Processing raw corpus file 'corpus.csv'"):
                doc_id = row.get("cid")
                if doc_id is None:
                    continue

                corpus[str(doc_id)] = normalize_text(row.get("text"))
                if max_docs is not None and len(corpus) >= max_docs:
                    break

    return corpus, {"split_name": "corpus.csv", "num_docs": len(corpus)}, raw_schema


def build_queries_and_qrels_from_eval_dataset(
    dataset,
    corpus_doc_ids: set[str],
    max_queries: int | None,
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int | float]]:
    queries = {}
    qrels = {}

    processed_rows = 0
    total_query_rows = 0
    missing_qid_count = 0
    duplicate_qid_count = 0
    total_relevant_doc_refs = 0
    found_relevant_doc_refs = 0
    missing_relevant_doc_refs = 0
    missing_unique_doc_ids = set()
    global_row_index = 0

    for split_name in dataset.keys():
        split = dataset[split_name]
        columns = set(split.column_names)

        if "question" not in columns or "cid" not in columns:
            continue

        for row_index, row in enumerate(tqdm(split, desc=f"Processing eval split '{split_name}'")):
            if max_queries is not None and processed_rows >= max_queries:
                found_rate = (
                    found_relevant_doc_refs / total_relevant_doc_refs
                    if total_relevant_doc_refs
                    else 0.0
                )
                return queries, qrels, {
                    "total_query_rows": total_query_rows,
                    "processed_queries": processed_rows,
                    "missing_qid_count": missing_qid_count,
                    "duplicate_qid_count": duplicate_qid_count,
                    "total_relevant_doc_refs": total_relevant_doc_refs,
                    "found_relevant_doc_refs": found_relevant_doc_refs,
                    "missing_relevant_doc_refs": missing_relevant_doc_refs,
                    "missing_unique_doc_ids": len(missing_unique_doc_ids),
                    "found_rate": found_rate,
                }

            total_query_rows += 1
            question = normalize_text(row.get("question")).strip()
            if not question:
                global_row_index += 1
                continue

            raw_qid = row.get("qid")
            if raw_qid is None:
                missing_qid_count += 1
                qid = str(global_row_index)
            else:
                qid = str(raw_qid)

            if qid in queries:
                duplicate_qid_count += 1
                qid = f"{qid}__dup_{duplicate_qid_count}"

            relevant_doc_ids = deduplicate_preserve_order(normalize_id_list(row.get("cid")))

            total_relevant_doc_refs += len(relevant_doc_ids)
            for doc_id in relevant_doc_ids:
                if doc_id in corpus_doc_ids:
                    found_relevant_doc_refs += 1
                else:
                    missing_relevant_doc_refs += 1
                    missing_unique_doc_ids.add(doc_id)

            queries[qid] = question
            qrels[qid] = relevant_doc_ids
            processed_rows += 1
            global_row_index += 1

    found_rate = (
        found_relevant_doc_refs / total_relevant_doc_refs
        if total_relevant_doc_refs
        else 0.0
    )
    return queries, qrels, {
        "total_query_rows": total_query_rows,
        "processed_queries": processed_rows,
        "missing_qid_count": missing_qid_count,
        "duplicate_qid_count": duplicate_qid_count,
        "total_relevant_doc_refs": total_relevant_doc_refs,
        "found_relevant_doc_refs": found_relevant_doc_refs,
        "missing_relevant_doc_refs": missing_relevant_doc_refs,
        "missing_unique_doc_ids": len(missing_unique_doc_ids),
        "found_rate": found_rate,
    }


def prepare_data(
    max_docs: int | None = None,
    max_queries: int | None = None,
    raw_dataset_name: str = DEFAULT_RAW_DATASET_NAME,
    eval_dataset_name: str = DEFAULT_EVAL_DATASET_NAME,
) -> dict[str, object]:
    print(f"Dang load raw dataset: {raw_dataset_name}")
    raw_dataset = None
    raw_schema = None

    if raw_dataset_name == DEFAULT_RAW_DATASET_NAME:
        print("Raw dataset BKAI se duoc doc truc tiep tu archive.zip de tach rieng corpus.csv.")
        corpus, corpus_stats, raw_schema = build_corpus_from_raw_archive(
            raw_dataset_name=raw_dataset_name,
            max_docs=max_docs,
        )
    else:
        try:
            raw_dataset = load_legal_dataset(raw_dataset_name)
            raw_schema = dataset_schema(raw_dataset)
            corpus, corpus_stats = build_corpus_from_raw_dataset(raw_dataset, max_docs=max_docs)
        except Exception as raw_exc:
            print(
                "Canh bao: khong the parse raw dataset bang datasets.load_dataset(). "
                "Se fallback sang doc truc tiep archive.zip."
            )
            print(f"Loi goc raw dataset: {raw_exc}")
            corpus, corpus_stats, raw_schema = build_corpus_from_raw_archive(
                raw_dataset_name=raw_dataset_name,
                max_docs=max_docs,
            )

    print(f"Dang load evaluation dataset: {eval_dataset_name}")
    eval_dataset = load_legal_dataset(eval_dataset_name)
    queries, qrels, eval_stats = build_queries_and_qrels_from_eval_dataset(
        eval_dataset,
        corpus_doc_ids=set(corpus.keys()),
        max_queries=max_queries,
    )

    save_json(corpus, CORPUS_PATH)
    save_json(queries, QUERIES_PATH)
    save_json(qrels, QRELS_PATH)

    print("\nHoan tat chuan bi du lieu.")
    print(f"Raw dataset schema: {raw_schema}")
    print(f"Eval dataset schema: {dataset_schema(eval_dataset)}")
    print(f"Corpus split duoc dung: {corpus_stats['split_name']}")
    print(f"So document trong corpus: {len(corpus)}")
    print(f"So query: {len(queries)}")
    print(f"So qrels: {len(qrels)}")
    print(f"So cid relevant tong cong: {eval_stats['total_relevant_doc_refs']}")
    print(f"So cid relevant tim thay trong corpus: {eval_stats['found_relevant_doc_refs']}")
    print(f"So cid relevant bi missing khoi corpus: {eval_stats['missing_relevant_doc_refs']}")
    print(f"So cid missing unique: {eval_stats['missing_unique_doc_ids']}")
    print(f"Ty le cid relevant tim thay trong corpus: {eval_stats['found_rate']:.2%}")
    print(f"So query khong co qid va da tu sinh: {eval_stats['missing_qid_count']}")
    if eval_stats["duplicate_qid_count"]:
        print(f"Canh bao: co {eval_stats['duplicate_qid_count']} qid bi trung, da them hau to __dup_.")
    if eval_stats["missing_relevant_doc_refs"]:
        print(
            "Warning: mot so cid trong qrels khong ton tai trong corpus raw. "
            "Van giu nguyen qrels de danh gia."
        )
        if max_docs is not None:
            print(
                "Ghi chu: canh bao nay co the do ban dang gioi han --max_docs, "
                "nen corpus.json hien tai chi la mot phan cua raw corpus."
            )
    print(f"Da luu: {CORPUS_PATH}")
    print(f"Da luu: {QUERIES_PATH}")
    print(f"Da luu: {QRELS_PATH}")

    return {
        "raw_dataset_name": raw_dataset_name,
        "eval_dataset_name": eval_dataset_name,
        "raw_schema": raw_schema,
        "eval_schema": dataset_schema(eval_dataset),
        "corpus_stats": corpus_stats,
        "eval_stats": eval_stats,
        "num_docs": len(corpus),
        "num_queries": len(queries),
        "num_qrels": len(qrels),
    }


def main() -> None:
    configure_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Chuan bi corpus tu raw dataset va queries/qrels tu evaluation dataset"
    )
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--raw_dataset_name", default=DEFAULT_RAW_DATASET_NAME)
    parser.add_argument("--eval_dataset_name", default=DEFAULT_EVAL_DATASET_NAME)
    args = parser.parse_args()

    prepare_data(
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        raw_dataset_name=args.raw_dataset_name,
        eval_dataset_name=args.eval_dataset_name,
    )


if __name__ == "__main__":
    main()
