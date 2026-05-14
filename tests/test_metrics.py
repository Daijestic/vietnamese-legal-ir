import pytest

from src.evaluation.metrics import (
    average_precision,
    dcg_at_k,
    evaluate_query,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    recall_at_k,
)


def test_precision_and_recall_at_k():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3", "d4"}

    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)


def test_precision_and_recall_without_k():
    retrieved = ["d1", "d2", "d3", "d5"]
    relevant = {"d1", "d3", "d4"}

    assert precision(retrieved, relevant) == pytest.approx(0.5)
    assert recall(retrieved, relevant) == pytest.approx(2 / 3)


def test_average_precision():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}

    assert average_precision(retrieved, relevant) == pytest.approx((1.0 + (2 / 3)) / 2)


def test_average_precision_with_cutoff():
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d1", "d3", "d4"}

    assert average_precision(retrieved, relevant, k=2) == pytest.approx(1 / 3)


def test_dcg_and_ndcg():
    retrieved = ["d1", "d2", "d3"]
    relevant = {"d1", "d3"}

    assert dcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0 + 1 / 2)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx((1.0 + 1 / 2) / (1.0 + 1 / 1.584962500721156))


def test_evaluate_query_handles_empty_inputs():
    metrics = evaluate_query([], set(), k_values=[5, 10])

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["average_precision"] == 0.0
    assert metrics["map"] == 0.0
    assert metrics["precision@5"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["ndcg@10"] == 0.0


def test_evaluate_query_handles_multiple_relevant_documents():
    retrieved = ["d3", "d1", "d5", "d2"]
    relevant = {"d1", "d2", "d4", "d5"}

    metrics = evaluate_query(retrieved, relevant, k_values=[5, 10])

    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["average_precision"] == pytest.approx(((1 / 2) + (2 / 3) + (3 / 4)) / 4)
