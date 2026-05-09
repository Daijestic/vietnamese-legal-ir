import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    corpus = load_json("data/processed/corpus.json")
    queries = load_json("data/processed/queries.json")
    qrels = load_json("data/processed/qrels.json")

    print("Số document:", len(corpus))
    print("Số query:", len(queries))
    print("Số qrels:", len(qrels))

    first_qid = next(iter(queries))
    print("\nQuery mẫu:")
    print("qid:", first_qid)
    print("question:", queries[first_qid])
    print("relevant cids:", qrels[first_qid])

    first_cid = qrels[first_qid][0]
    print("\nDocument liên quan mẫu:")
    print("cid:", first_cid)
    print(corpus.get(first_cid, "")[:500])


if __name__ == "__main__":
    main()