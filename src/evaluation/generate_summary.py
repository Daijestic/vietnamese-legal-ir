import argparse
import csv
from pathlib import Path


DEFAULT_REPORT_PATH = Path("outputs/reports/evaluation_report.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/reports/summary.md")


def read_report(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def best_method(rows: list[dict[str, str]], metric: str) -> dict[str, str]:
    return max(rows, key=lambda row: float(row.get(metric, 0.0)))


def build_summary(rows: list[dict[str, str]]) -> str:
    best_map = best_method(rows, "map")
    best_ndcg = best_method(rows, "ndcg@10")

    lines = [
        "# Evaluation Summary",
        "",
        f"- Best MAP: **{best_map['method']}** ({float(best_map['map']):.4f})",
        f"- Best nDCG@10: **{best_ndcg['method']}** ({float(best_ndcg['ndcg@10']):.4f})",
        "",
        "| Method | Precision@5 | Recall@5 | MAP | nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| {method} | {p5:.4f} | {r5:.4f} | {map_score:.4f} | {ndcg10:.4f} |".format(
                method=row["method"],
                p5=float(row.get("precision@5", 0.0)),
                r5=float(row.get("recall@5", 0.0)),
                map_score=float(row.get("map", 0.0)),
                ndcg10=float(row.get("ndcg@10", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Nhan xet ngan",
            "",
            "- Boolean Retrieval: nhanh va de giai thich, nhung phu thuoc nhieu vao khop tu khoa chinh xac.",
            "- TF-IDF: la baseline ranking ro rang, can bang giua do don gian va kha nang xep hang.",
            "- BM25: thuong on dinh hon tren van ban dai nho co che chuan hoa do dai tai lieu.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Sinh summary markdown tu evaluation_report.csv")
    parser.add_argument("--report_path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary_path", default=str(DEFAULT_SUMMARY_PATH))
    args = parser.parse_args()

    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)

    if not report_path.exists():
        print(f"Khong tim thay file bao cao: {report_path}")
        print("Hay chay evaluation truoc khi sinh summary.")
        return

    rows = read_report(report_path)
    if not rows:
        print(f"File bao cao rong: {report_path}")
        return

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(rows), encoding="utf-8")
    print(f"Da luu summary tai: {summary_path}")


if __name__ == "__main__":
    main()
