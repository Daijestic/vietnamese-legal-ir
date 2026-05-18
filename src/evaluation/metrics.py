import math


def precision(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved or not relevant:
        return 0.0

    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / len(retrieved)


def recall(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    if not retrieved:
        return 0.0

    hits = sum(1 for doc_id in retrieved if doc_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return recall(retrieved[:k], relevant)


def average_precision(retrieved: list[str], relevant: set[str], k: int | None = None) -> float:
    if not relevant:
        return 0.0

    ranked_docs = retrieved if k is None else retrieved[:k]
    if not ranked_docs:
        return 0.0

    precision_sum = 0.0
    hit_count = 0
    seen = set()

    for rank, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in seen:
            continue
        seen.add(doc_id)

        if doc_id not in relevant:
            continue

        hit_count += 1
        precision_sum += hit_count / rank

    if hit_count == 0:
        return 0.0

    return precision_sum / len(relevant)


def dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0

    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        gain = 1.0 if doc_id in relevant else 0.0
        if rank == 1:
            score += gain
        else:
            score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(retrieved, relevant, k)
    ideal_hits = min(len(relevant), k)
    ideal_ranking = [f"rel_{idx}" for idx in range(ideal_hits)]
    ideal_relevant = set(ideal_ranking)
    ideal_dcg = dcg_at_k(ideal_ranking, ideal_relevant, k)
    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(
    retrieved_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 10,
) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
        f"recall@{k}": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
        "map": average_precision(retrieved_doc_ids, relevant_doc_ids),
        f"ndcg@{k}": ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k),
    }


if __name__ == "__main__":
    retrieved = ["d1", "d2", "d3", "d4"]
    relevant = {"d1", "d3"}
    print(evaluate_query(retrieved, relevant, k=10))
