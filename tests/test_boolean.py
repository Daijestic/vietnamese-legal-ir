from src.indexing.inverted_index import InvertedIndex
from src.retrieval.boolean_retrieval import BooleanRetriever


def test_inverted_index_postings():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }

    index = InvertedIndex()
    index.build(corpus)

    postings = index.get_postings("kết hôn")

    assert "1" in postings


def test_boolean_search_and():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }

    index = InvertedIndex()
    index.build(corpus)

    retriever = BooleanRetriever(index)
    results = retriever.search("điều kiện kết hôn", top_k=5)

    doc_ids = [item["doc_id"] for item in results]

    assert "1" in doc_ids


def test_boolean_search_or():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }

    index = InvertedIndex()
    index.build(corpus)

    retriever = BooleanRetriever(index)
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
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật",
        "2": "Hợp đồng lao động",
        "3": "Bảo hiểm xã hội",
    }

    index = InvertedIndex()
    index.build(corpus)

    retriever = BooleanRetriever(index)
    results = retriever.search("kết hôn or lao động", top_k=5)

    doc_ids = [item["doc_id"] for item in results]

    assert "1" in doc_ids
    assert "2" in doc_ids