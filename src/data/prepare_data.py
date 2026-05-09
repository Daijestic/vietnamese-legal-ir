import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "YuITC/Vietnamese-Legal-Documents"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_cid_list(cid_value):
    """
    Dataset có cột cid là list.
    Hàm này ép cid về list string.
    """
    if cid_value is None:
        return []

    if isinstance(cid_value, list):
        return [str(x) for x in cid_value]

    return [str(cid_value)]


def normalize_context_list(context_value):
    """
    Dataset có cột context_list là list text.
    """
    if context_value is None:
        return []

    if isinstance(context_value, list):
        return [str(x) for x in context_value]

    return [str(context_value)]


def prepare_data(max_queries=None, max_docs=None):
    print(f"Đang load dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    corpus = {}
    queries = {}
    qrels = {}

    total_queries = 0

    for split_name in ds.keys():
        print(f"Đang xử lý split: {split_name}")

        for row in tqdm(ds[split_name], desc=f"Processing {split_name}"):
            if max_queries is not None and total_queries >= max_queries:
                break

            qid = str(row.get("qid"))
            question = row.get("question", "")

            cid_list = normalize_cid_list(row.get("cid"))
            context_list = normalize_context_list(row.get("context_list"))

            if not qid or qid == "None":
                continue

            queries[qid] = str(question)
            qrels[qid] = cid_list
            total_queries += 1

            # Ghép cid với context tương ứng để tạo corpus
            for cid, context in zip(cid_list, context_list):
                if max_docs is not None and len(corpus) >= max_docs:
                    continue

                if cid not in corpus:
                    corpus[cid] = context

        if max_queries is not None and total_queries >= max_queries:
            break

    save_json(corpus, "data/processed/corpus.json")
    save_json(queries, "data/processed/queries.json")
    save_json(qrels, "data/processed/qrels.json")

    print("\nHoàn tất chuẩn hóa dữ liệu.")
    print(f"Số document trong corpus: {len(corpus)}")
    print(f"Số query: {len(queries)}")
    print(f"Số qrels: {len(qrels)}")
    print("Đã lưu:")
    print("- data/processed/corpus.json")
    print("- data/processed/queries.json")
    print("- data/processed/qrels.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)
    args = parser.parse_args()

    prepare_data(max_queries=args.max_queries, max_docs=args.max_docs)


if __name__ == "__main__":
    main()