from src.indexing.inverted_index import InvertedIndex
from src.retrieval.boolean_retrieval import BooleanRetriever


def build_sample_index() -> InvertedIndex:
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Nguoi lao dong duoc nghi hang nam",
        "3": "Quy dinh ve hop dong lao dong",
    }
    index = InvertedIndex()
    index.build(corpus)
    return index


def test_inverted_index_postings():
    index = build_sample_index()
    postings = index.get_postings("ket hon")
    assert "1" in postings


def test_boolean_search_and():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("dieu kien ket hon", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids


def test_boolean_search_or():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("ket hon OR lao dong", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids
    assert "2" in doc_ids


def test_boolean_search_not():
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Ly hon va chia tai san chung",
        "3": "Quy dinh ve hop dong lao dong",
    }
    index = InvertedIndex()
    index.build(corpus)
    retriever = BooleanRetriever(index)

    results = retriever.search("ket hon NOT ly", top_k=5)
    doc_ids = [item["doc_id"] for item in results]

    assert "1" in doc_ids
    assert "2" not in doc_ids


def test_boolean_search_or_lowercase():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("ket hon or lao dong", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids
    assert "2" in doc_ids


def test_boolean_search_respects_top_k():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("ket hon OR lao dong", top_k=2)

    assert len(results) == 2
    assert [item["doc_id"] for item in results] == ["1", "2"]


def test_inverted_index_save_pickle_and_load(tmp_path):
    index = build_sample_index()
    pickle_path = tmp_path / "inverted_index.pkl"
    index.save_pickle(pickle_path)

    loaded = InvertedIndex.load_pickle(pickle_path)

    assert "1" in loaded.get_postings("ket hon")
