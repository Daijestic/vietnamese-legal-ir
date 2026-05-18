from src.retrieval.bm25_retrieval import BM25Retriever


def test_bm25_ranks_relevant_document_higher():
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Nguoi lao dong duoc nghi hang nam",
        "3": "Quy dinh ve hop dong lao dong",
    }

    retriever = BM25Retriever()
    retriever.fit(corpus)

    results = retriever.search("dieu kien ket hon", top_k=3)

    assert results
    assert results[0]["doc_id"] == "1"
    assert all(item["score"] > 0 for item in results)


def test_bm25_returns_empty_for_unknown_query():
    retriever = BM25Retriever()
    retriever.fit({"1": "hop dong lao dong"})

    results = retriever.search("thua ke dat dai", top_k=3)

    assert results == []


def test_bm25_respects_top_k():
    corpus = {
        "1": "dieu kien ket hon ket hon",
        "2": "dieu kien ket hon",
        "3": "lao dong va hop dong",
    }
    retriever = BM25Retriever()
    retriever.fit(corpus)

    results = retriever.search("dieu kien ket hon", top_k=1)

    assert len(results) == 1
    assert results[0]["doc_id"] == "1"


def test_bm25_save_and_load(tmp_path):
    corpus = {
        "1": "Dieu kien ket hon theo phap luat Viet Nam",
        "2": "Nguoi lao dong duoc nghi hang nam",
    }

    retriever = BM25Retriever()
    retriever.fit(corpus)

    model_path = tmp_path / "bm25_model.pkl"
    retriever.save(model_path)
    loaded = BM25Retriever.load(model_path)

    results = loaded.search("dieu kien ket hon", top_k=2)
    assert results
    assert results[0]["doc_id"] == "1"
