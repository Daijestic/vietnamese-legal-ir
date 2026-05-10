from src.indexing.inverted_index import InvertedIndex
from src.retrieval.boolean_retrieval import BooleanRetriever


def build_sample_index() -> InvertedIndex:
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }
    index = InvertedIndex()
    index.build(corpus)
    return index


def test_inverted_index_postings():
    index = build_sample_index()
    postings = index.get_postings("kết hôn")
    assert "1" in postings


def test_boolean_search_and():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("điều kiện kết hôn", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids


def test_boolean_search_or():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("kết hôn OR lao động", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids
    assert "2" in doc_ids


def test_boolean_search_not():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Ly hôn và chia tài sản chung",
        "3": "Quy định về hợp đồng lao động",
    }
    index = InvertedIndex()
    index.build(corpus)
    retriever = BooleanRetriever(index)

    results = retriever.search("kết hôn NOT ly", top_k=5)
    doc_ids = [item["doc_id"] for item in results]

    assert "1" in doc_ids
    assert "2" not in doc_ids


def test_boolean_search_or_lowercase():
    retriever = BooleanRetriever(build_sample_index())
    results = retriever.search("kết hôn or lao động", top_k=5)
    doc_ids = [item["doc_id"] for item in results]
    assert "1" in doc_ids
    assert "2" in doc_ids


def test_inverted_index_save_pickle_and_load(tmp_path):
    index = build_sample_index()
    pickle_path = tmp_path / "inverted_index.pkl"
    index.save_pickle(pickle_path)

    loaded = InvertedIndex.load_pickle(pickle_path)

    assert "1" in loaded.get_postings("kết hôn")
