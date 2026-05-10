from src.retrieval.tfidf_retrieval import TfidfRetriever


def test_tfidf_ranks_relevant_document_higher():
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
        "3": "Quy định về hợp đồng lao động",
    }

    retriever = TfidfRetriever()
    retriever.fit(corpus)

    results = retriever.search("điều kiện kết hôn", top_k=3)

    assert results
    assert results[0]["doc_id"] == "1"
    assert all(item["score"] > 0 for item in results)


def test_tfidf_returns_empty_for_unknown_query():
    retriever = TfidfRetriever()
    retriever.fit({"1": "hợp đồng lao động"})

    results = retriever.search("thừa kế đất đai", top_k=3)

    assert results == []


def test_tfidf_save_and_load(tmp_path):
    corpus = {
        "1": "Điều kiện kết hôn theo pháp luật Việt Nam",
        "2": "Người lao động được nghỉ hằng năm",
    }

    retriever = TfidfRetriever()
    retriever.fit(corpus)

    model_path = tmp_path / "tfidf_model.pkl"
    retriever.save(model_path)
    loaded = TfidfRetriever.load(model_path)

    results = loaded.search("điều kiện kết hôn", top_k=2)
    assert results
    assert results[0]["doc_id"] == "1"
