import argparse
import json
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from src.preprocessing.tokenizer import preprocess


class InvertedIndex:
    """
    Inverted Index cho Boolean Retrieval.

    postings:
        token -> sorted list doc_id

    doc_lengths:
        doc_id -> số token sau preprocessing

    document_frequency:
        token -> số document chứa token
    """

    def __init__(self):
        self.postings = defaultdict(set)
        self.doc_lengths = {}
        self.document_frequency = {}
        self.vocabulary = set()
        self.num_docs = 0

    def build(self, corpus: dict[str, str]) -> None:
        self.num_docs = len(corpus)

        for doc_id, text in tqdm(corpus.items(), desc="Building inverted index"):
            doc_id = str(doc_id)
            tokens = preprocess(text)
            self.doc_lengths[doc_id] = len(tokens)

            unique_tokens = set(tokens)

            for token in unique_tokens:
                self.postings[token].add(doc_id)

        self.vocabulary = set(self.postings.keys())
        self.document_frequency = {
            token: len(doc_ids)
            for token, doc_ids in self.postings.items()
        }

        self.postings = {
            token: sorted(list(doc_ids))
            for token, doc_ids in self.postings.items()
        }

    def get_postings(self, term: str) -> list[str]:
        tokens = preprocess(term)

        if not tokens:
            return []

        if len(tokens) == 1:
            return self.postings.get(tokens[0], [])

        # Nếu người dùng truyền cụm từ, lấy AND giữa các token
        result = set(self.postings.get(tokens[0], []))

        for token in tokens[1:]:
            result &= set(self.postings.get(token, []))

        return sorted(result)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "postings": self.postings,
            "doc_lengths": self.doc_lengths,
            "document_frequency": self.document_frequency,
            "vocabulary": sorted(list(self.vocabulary)),
            "num_docs": self.num_docs,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        index = cls()
        index.postings = data.get("postings", {})
        index.doc_lengths = data.get("doc_lengths", {})
        index.document_frequency = data.get("document_frequency", {})
        index.vocabulary = set(data.get("vocabulary", []))
        index.num_docs = data.get("num_docs", len(index.doc_lengths))

        return index


def load_corpus(path: str | Path) -> dict[str, str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy: python src/data/prepare_data.py trước."
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build inverted index")
    parser.add_argument(
        "--corpus_path",
        default="data/processed/corpus.json",
        help="Đường dẫn corpus.json",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/indexes/inverted_index.json",
        help="Đường dẫn lưu inverted index",
    )

    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)

    index = InvertedIndex()
    index.build(corpus)
    index.save(args.output_path)

    print(f"Đã build inverted index.")
    print(f"Số document: {index.num_docs}")
    print(f"Kích thước vocabulary: {len(index.vocabulary)}")
    print(f"Lưu tại: {args.output_path}")


if __name__ == "__main__":
    main()