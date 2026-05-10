import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.preprocessing.tokenizer import preprocess


class InvertedIndex:
    """
    Inverted index don gian cho Boolean Retrieval.

    postings:
        token -> sorted list[doc_id]

    doc_lengths:
        doc_id -> so token sau preprocessing

    document_frequency:
        token -> so document chua token
    """

    def __init__(self):
        self.postings = defaultdict(set)
        self.doc_lengths = {}
        self.document_frequency = {}
        self.vocabulary = set()
        self.num_docs = 0

    def build(self, corpus: dict[str, str]) -> None:
        self.postings = defaultdict(set)
        self.doc_lengths = {}
        self.document_frequency = {}
        self.vocabulary = set()
        self.num_docs = len(corpus)

        for doc_id, text in tqdm(corpus.items(), desc="Building inverted index"):
            doc_id = str(doc_id)
            tokens = preprocess(text)
            self.doc_lengths[doc_id] = len(tokens)

            for token in set(tokens):
                self.postings[token].add(doc_id)

        self.vocabulary = set(self.postings.keys())
        self.document_frequency = {
            token: len(doc_ids)
            for token, doc_ids in self.postings.items()
        }
        self.postings = {
            token: sorted(doc_ids)
            for token, doc_ids in self.postings.items()
        }

    def get_postings(self, term: str) -> list[str]:
        query_tokens = preprocess(term)

        if not query_tokens:
            return []

        if len(query_tokens) == 1:
            return list(self.postings.get(query_tokens[0], []))

        candidate_docs = set(self.postings.get(query_tokens[0], []))
        for token in query_tokens[1:]:
            candidate_docs &= set(self.postings.get(token, []))

        return sorted(candidate_docs)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "postings": dict(self.postings),
            "doc_lengths": self.doc_lengths,
            "document_frequency": self.document_frequency,
            "vocabulary": sorted(self.vocabulary),
            "num_docs": self.num_docs,
        }

        if output_path.suffix.lower() == ".pkl":
            with output_path.open("wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
            return

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def save_pickle(self, path: str | Path) -> None:
        self.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        input_path = Path(path)
        if input_path.suffix.lower() == ".pkl":
            with input_path.open("rb") as file:
                data = pickle.load(file)
        else:
            with input_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        index = cls()
        index.postings = data.get("postings", {})
        index.doc_lengths = data.get("doc_lengths", {})
        index.document_frequency = data.get("document_frequency", {})
        index.vocabulary = set(data.get("vocabulary", []))
        index.num_docs = data.get("num_docs", len(index.doc_lengths))
        return index

    @classmethod
    def load_pickle(cls, path: str | Path) -> "InvertedIndex":
        return cls.load(path)


def load_corpus(path: str | Path) -> dict[str, str]:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay {corpus_path}. Hay chay: python -m src.data.prepare_data truoc."
        )

    with corpus_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser(description="Build inverted index")
    parser.add_argument(
        "--corpus_path",
        default="data/processed/corpus.json",
        help="Duong dan den corpus.json",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/indexes/inverted_index.json",
        help="Duong dan luu inverted index",
    )
    parser.add_argument(
        "--pickle_path",
        default="outputs/indexes/inverted_index.pkl",
        help="Duong dan luu inverted index pickle",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)

    index = InvertedIndex()
    index.build(corpus)
    index.save(args.output_path)
    index.save_pickle(args.pickle_path)

    print("Da build inverted index.")
    print(f"So document: {index.num_docs}")
    print(f"Kich thuoc vocabulary: {len(index.vocabulary)}")
    print(f"Luu tai: {args.output_path}")
    print(f"Luu tai: {args.pickle_path}")


if __name__ == "__main__":
    main()
