from src.retrieval.bm25_retrieval import BM25Retriever


def test_bm25_ranks_relevant_document_higher():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }

    retriever = BM25Retriever()
    retriever.fit(corpus)

    results = retriever.search("điều kiện kết hôn", top_k=3)

    assert results
    assert results[0]["doc_id"] == "1"
    assert all(item["score"] > 0 for item in results)


def test_bm25_returns_empty_for_unknown_query():
    retriever = BM25Retriever()
    retriever.fit({"1": "hợp đồng lao động"})

    results = retriever.search("thừa kế đất đai", top_k=3)

    assert results == []


def test_bm25_save_and_load(tmp_path):
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
    }

    retriever = BM25Retriever()
    retriever.fit(corpus)

    model_path = tmp_path / "bm25_model.pkl"
    retriever.save(model_path)
    loaded = BM25Retriever.load(model_path)

    results = loaded.search("điều kiện kết hôn", top_k=2)
    assert results
    assert results[0]["doc_id"] == "1"
