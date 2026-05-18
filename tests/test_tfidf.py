from src.retrieval.tfidf_retrieval import TfidfRetriever


def test_tfidf_ranks_relevant_document_higher():
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Nguoi lao dong duoc nghi hang nam",
        "3": "Quy dinh ve hop dong lao dong",
    }

    retriever = TfidfRetriever()
    retriever.fit(corpus)

    results = retriever.search("dieu kien ket hon", top_k=3)

    assert results
    assert results[0]["doc_id"] == "1"
    assert all(item["score"] > 0 for item in results)


def test_tfidf_returns_empty_for_unknown_query():
    retriever = TfidfRetriever()
    retriever.fit({"1": "hop dong lao dong"})

    results = retriever.search("thua ke dat dai", top_k=3)

    assert results == []


def test_tfidf_respects_top_k():
    corpus = {
        "1": "dieu kien ket hon ket hon",
        "2": "dieu kien ket hon",
        "3": "lao dong va hop dong",
    }
    retriever = TfidfRetriever()
    retriever.fit(corpus)

    results = retriever.search("dieu kien ket hon", top_k=1)

    assert len(results) == 1
    assert results[0]["doc_id"] in {"1", "2"}


def test_tfidf_save_and_load(tmp_path):
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Nguoi lao dong duoc nghi hang nam",
    }

    retriever = TfidfRetriever()
    retriever.fit(corpus)

    model_path = tmp_path / "tfidf_model.pkl"
    retriever.save(model_path)
    loaded = TfidfRetriever.load(model_path)

    results = loaded.search("dieu kien ket hon", top_k=2)
    assert results
    assert results[0]["doc_id"] == "1"
