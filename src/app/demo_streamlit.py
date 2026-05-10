import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retrieval import BM25Retriever
from src.retrieval.boolean_retrieval import BooleanRetriever
from src.retrieval.tfidf_retrieval import TfidfRetriever


st.set_page_config(
    page_title="Vietnamese Legal Document Retrieval Demo",
    page_icon="⚖️",
    layout="wide",
)


CORPUS_PATH = ROOT_DIR / "data" / "processed" / "corpus.json"
QUERIES_PATH = ROOT_DIR / "data" / "processed" / "queries.json"
QRELS_PATH = ROOT_DIR / "data" / "processed" / "qrels.json"
INDEX_JSON_PATH = ROOT_DIR / "outputs" / "indexes" / "inverted_index.json"
INDEX_PKL_PATH = ROOT_DIR / "outputs" / "indexes" / "inverted_index.pkl"
TFIDF_MODEL_PATH = ROOT_DIR / "outputs" / "indexes" / "tfidf_model.pkl"
BM25_MODEL_PATH = ROOT_DIR / "outputs" / "indexes" / "bm25_model.pkl"
EVALUATION_PATH = ROOT_DIR / "outputs" / "reports" / "evaluation_report.csv"
SUMMARY_PATH = ROOT_DIR / "outputs" / "reports" / "summary.md"

SAMPLE_QUERIES = [
    "điều kiện kết hôn theo pháp luật Việt Nam",
    "người lao động được nghỉ hằng năm bao nhiêu ngày",
    "hợp đồng lao động bị chấm dứt trong trường hợp nào",
    "quyền nuôi con sau khi ly hôn",
    "xử phạt vi phạm giao thông đường bộ",
]

METHOD_LABELS = {
    "boolean": "Boolean Retrieval",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
}

COMPARISON_PREFIX = {
    "boolean": "Boolean",
    "tfidf": "TF-IDF",
    "bm25": "BM25",
}


def file_signature(path: Path) -> tuple[bool, float, int]:
    if not path.exists():
        return (False, 0.0, 0)
    stat = path.stat()
    return (True, stat.st_mtime, stat.st_size)


def get_preferred_index_path() -> Path | None:
    if INDEX_PKL_PATH.exists():
        return INDEX_PKL_PATH
    if INDEX_JSON_PATH.exists():
        return INDEX_JSON_PATH
    return None


def retrieval_artifacts_ready() -> bool:
    return (
        get_preferred_index_path() is not None
        and TFIDF_MODEL_PATH.exists()
        and BM25_MODEL_PATH.exists()
    )


def shorten(text: str, max_chars: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars].rstrip() + "..."


@st.cache_data(show_spinner=False)
def load_json_file(path_str: str, signature: tuple[bool, float, int]):
    path = Path(path_str)
    if not signature[0]:
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def load_markdown_file(path_str: str, signature: tuple[bool, float, int]) -> str:
    path = Path(path_str)
    if not signature[0]:
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_evaluation_frame(path_str: str, signature: tuple[bool, float, int]) -> pd.DataFrame:
    path = Path(path_str)
    if not signature[0]:
        raise FileNotFoundError(path)
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_boolean_retriever(index_path_str: str, signature: tuple[bool, float, int]) -> BooleanRetriever:
    index_path = Path(index_path_str)
    if not signature[0]:
        raise FileNotFoundError(
            "Chưa có retrieval models. Hãy chạy: python -m src.indexing.build_retrieval_models"
        )
    return BooleanRetriever(InvertedIndex.load(index_path))


@st.cache_resource(show_spinner=False)
def get_tfidf_retriever(model_path_str: str, signature: tuple[bool, float, int]) -> TfidfRetriever:
    model_path = Path(model_path_str)
    if not signature[0]:
        raise FileNotFoundError(
            "Chưa có retrieval models. Hãy chạy: python -m src.indexing.build_retrieval_models"
        )
    return TfidfRetriever.load(model_path)


@st.cache_resource(show_spinner=False)
def get_bm25_retriever(model_path_str: str, signature: tuple[bool, float, int]) -> BM25Retriever:
    model_path = Path(model_path_str)
    if not signature[0]:
        raise FileNotFoundError(
            "Chưa có retrieval models. Hãy chạy: python -m src.indexing.build_retrieval_models"
        )
    return BM25Retriever.load(model_path)


def render_status_item(path: Path, label: str) -> None:
    if path.exists():
        st.sidebar.success(f"{label}: OK")
    else:
        st.sidebar.error(f"{label}: Missing")


def render_index_status() -> None:
    if INDEX_PKL_PATH.exists():
        st.sidebar.success("outputs/indexes/inverted_index.pkl: OK")
    elif INDEX_JSON_PATH.exists():
        st.sidebar.warning("outputs/indexes/inverted_index.pkl: Missing")
        st.sidebar.success("outputs/indexes/inverted_index.json: OK (fallback)")
    else:
        st.sidebar.error("outputs/indexes/inverted_index.pkl: Missing")
        st.sidebar.error("outputs/indexes/inverted_index.json: Missing")


def run_search(method_key: str, query: str, top_k: int):
    start_time = time.perf_counter()

    if method_key == "boolean":
        index_path = get_preferred_index_path()
        if index_path is None:
            raise FileNotFoundError(
                "Chưa có retrieval models. Hãy chạy: python -m src.indexing.build_retrieval_models"
            )
        retriever = get_boolean_retriever(str(index_path), file_signature(index_path))
    elif method_key == "tfidf":
        retriever = get_tfidf_retriever(str(TFIDF_MODEL_PATH), file_signature(TFIDF_MODEL_PATH))
    elif method_key == "bm25":
        retriever = get_bm25_retriever(str(BM25_MODEL_PATH), file_signature(BM25_MODEL_PATH))
    else:
        raise ValueError(f"Unsupported method: {method_key}")

    results = retriever.search(query, top_k=top_k)
    elapsed = time.perf_counter() - start_time
    return results, elapsed


def results_to_frame(results: list[dict], corpus: dict[str, str], snippet_chars: int) -> pd.DataFrame:
    rows = []
    for rank, item in enumerate(results, start=1):
        doc_id = str(item.get("doc_id", ""))
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": float(item.get("score", 0.0)),
                "method": item.get("method", ""),
                "snippet": shorten(corpus.get(doc_id, ""), snippet_chars),
            }
        )
    return pd.DataFrame(rows)


def build_comparison_frame(results_by_method: dict[str, list[dict]], top_k: int) -> pd.DataFrame:
    rows = []
    for rank in range(1, top_k + 1):
        row = {"Rank": rank}
        for method_key in ("boolean", "tfidf", "bm25"):
            method_results = results_by_method.get(method_key, [])
            item = method_results[rank - 1] if rank - 1 < len(method_results) else {}
            prefix = COMPARISON_PREFIX[method_key]
            row[f"{prefix} doc_id"] = item.get("doc_id", "")
            row[f"{prefix} score"] = round(float(item.get("score", 0.0)), 6) if item else None
        rows.append(row)
    return pd.DataFrame(rows)


def render_result_tab(
    tab,
    results: list[dict],
    corpus: dict[str, str],
    snippet_chars: int,
    show_full_text: bool,
    elapsed: float | None,
    error_message: str | None,
) -> None:
    with tab:
        if error_message:
            st.error(error_message)
            return

        if elapsed is not None:
            st.caption(f"Search time: {elapsed:.4f}s")

        if not results:
            st.info("Không có kết quả phù hợp cho truy vấn này.")
            return

        st.dataframe(
            results_to_frame(results, corpus, snippet_chars),
            use_container_width=True,
            hide_index=True,
        )

        for rank, item in enumerate(results, start=1):
            doc_id = str(item["doc_id"])
            st.markdown(
                f"**#{rank} | doc_id:** `{doc_id}` | **score:** `{item['score']:.6f}` | **method:** `{item['method']}`"
            )

            with st.expander(f"Xem tài liệu {doc_id}"):
                snippet = shorten(corpus.get(doc_id, ""), snippet_chars)
                st.write(snippet if snippet else "Không có nội dung văn bản trong corpus.")
                if show_full_text:
                    st.divider()
                    full_text = corpus.get(doc_id, "")
                    st.write(full_text if full_text else "Không có nội dung đầy đủ.")


def render_intersection_section(results_by_method: dict[str, list[dict]]) -> None:
    boolean_docs = {str(item["doc_id"]) for item in results_by_method.get("boolean", [])}
    tfidf_docs = {str(item["doc_id"]) for item in results_by_method.get("tfidf", [])}
    bm25_docs = {str(item["doc_id"]) for item in results_by_method.get("bm25", [])}

    intersections = {
        "Boolean ∩ TF-IDF": sorted(boolean_docs & tfidf_docs),
        "Boolean ∩ BM25": sorted(boolean_docs & bm25_docs),
        "TF-IDF ∩ BM25": sorted(tfidf_docs & bm25_docs),
        "Boolean ∩ TF-IDF ∩ BM25": sorted(boolean_docs & tfidf_docs & bm25_docs),
    }

    cols = st.columns(4)
    for col, (label, docs) in zip(cols, intersections.items()):
        with col:
            st.metric(label, len(docs))
            st.caption(", ".join(docs[:8]) if docs else "Không có")


def render_evaluation_section() -> None:
    st.subheader("Evaluation")

    evaluation_signature = file_signature(EVALUATION_PATH)
    summary_signature = file_signature(SUMMARY_PATH)

    if evaluation_signature[0]:
        frame = load_evaluation_frame(str(EVALUATION_PATH), evaluation_signature)
        st.dataframe(frame, use_container_width=True, hide_index=True)

        chart_columns = []
        if "map" in frame.columns:
            chart_columns.append("map")
        if "ndcg@10" in frame.columns:
            chart_columns.append("ndcg@10")

        if chart_columns and "method" in frame.columns:
            st.bar_chart(frame.set_index("method")[chart_columns])
    else:
        st.info(
            "Chưa có file evaluation. Hãy chạy: python -m src.evaluation.evaluate --method all --top_k 10"
        )

    if summary_signature[0]:
        with st.expander("Xem summary.md"):
            st.markdown(load_markdown_file(str(SUMMARY_PATH), summary_signature))


def main() -> None:
    st.title("Vietnamese Legal Document Retrieval Demo")
    st.write(
        "Demo hệ thống truy xuất văn bản pháp luật tiếng Việt sử dụng Boolean Retrieval, TF-IDF và BM25."
    )

    st.sidebar.header("Cấu hình")
    render_status_item(CORPUS_PATH, "data/processed/corpus.json")
    render_status_item(QUERIES_PATH, "data/processed/queries.json")
    render_status_item(QRELS_PATH, "data/processed/qrels.json")
    render_index_status()
    render_status_item(TFIDF_MODEL_PATH, "outputs/indexes/tfidf_model.pkl")
    render_status_item(BM25_MODEL_PATH, "outputs/indexes/bm25_model.pkl")

    top_k = st.sidebar.selectbox("Top-K", options=[5, 10, 15, 20], index=0)
    show_full_text = st.sidebar.checkbox("Hiển thị toàn văn", value=False)
    snippet_chars = st.sidebar.slider(
        "Số ký tự snippet",
        min_value=150,
        max_value=1200,
        value=400,
        step=50,
    )

    st.sidebar.subheader("Chuẩn bị full data")
    st.sidebar.code(
        "python -m src.data.prepare_data\npython -m src.indexing.build_retrieval_models",
        language="bash",
    )

    corpus_signature = file_signature(CORPUS_PATH)
    if not corpus_signature[0]:
        st.error("Chưa có dữ liệu processed. Hãy chạy: python -m src.data.prepare_data")
        st.stop()

    if not retrieval_artifacts_ready():
        st.error("Chưa có retrieval models. Hãy chạy: python -m src.indexing.build_retrieval_models")
        st.stop()

    try:
        corpus = load_json_file(str(CORPUS_PATH), corpus_signature)
    except Exception as exc:
        st.error(f"Không thể load corpus.json: {exc}")
        st.stop()

    st.subheader("Truy vấn")
    selected_query = st.selectbox("Query mẫu", options=[""] + SAMPLE_QUERIES, index=0)

    if "query_input" not in st.session_state:
        st.session_state["query_input"] = selected_query
    if st.session_state.get("_last_selected_query") != selected_query and selected_query:
        st.session_state["query_input"] = selected_query
    st.session_state["_last_selected_query"] = selected_query

    manual_query = st.text_area(
        "Nhập câu hỏi pháp luật tiếng Việt",
        key="query_input",
        height=100,
        placeholder="Ví dụ: điều kiện kết hôn theo pháp luật Việt Nam",
    )

    search_clicked = st.button(
        "Tìm kiếm và so sánh 3 phương pháp",
        type="primary",
        use_container_width=True,
    )

    if search_clicked:
        query = manual_query.strip()
        if not query:
            st.warning("Vui lòng nhập query.")
            st.stop()

        results_by_method = {}
        errors_by_method = {}
        timings = {}
        summary_rows = []

        with st.spinner("Đang load artifact đã cache và chạy 3 phương pháp..."):
            for method_key in ("boolean", "tfidf", "bm25"):
                try:
                    results, elapsed = run_search(method_key, query, top_k)
                    results_by_method[method_key] = results
                    timings[method_key] = elapsed
                    summary_rows.append(
                        {
                            "Method": METHOD_LABELS[method_key],
                            "Num Results": len(results),
                            "Search Time (s)": round(elapsed, 6),
                            "Best Score": round(float(results[0]["score"]), 6) if results else None,
                        }
                    )
                except Exception as exc:
                    results_by_method[method_key] = []
                    errors_by_method[method_key] = str(exc)
                    timings[method_key] = None
                    summary_rows.append(
                        {
                            "Method": METHOD_LABELS[method_key],
                            "Num Results": 0,
                            "Search Time (s)": None,
                            "Best Score": None,
                        }
                    )

        st.subheader("Tổng quan")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if errors_by_method:
            for method_key, error_message in errors_by_method.items():
                st.warning(f"{METHOD_LABELS[method_key]} lỗi: {error_message}")

        tabs = st.tabs(["Boolean Retrieval", "TF-IDF", "BM25"])
        render_result_tab(
            tabs[0],
            results_by_method["boolean"],
            corpus,
            snippet_chars,
            show_full_text,
            timings["boolean"],
            errors_by_method.get("boolean"),
        )
        render_result_tab(
            tabs[1],
            results_by_method["tfidf"],
            corpus,
            snippet_chars,
            show_full_text,
            timings["tfidf"],
            errors_by_method.get("tfidf"),
        )
        render_result_tab(
            tabs[2],
            results_by_method["bm25"],
            corpus,
            snippet_chars,
            show_full_text,
            timings["bm25"],
            errors_by_method.get("bm25"),
        )

        st.subheader("Bảng so sánh kết quả")
        st.dataframe(
            build_comparison_frame(results_by_method, top_k),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Tài liệu trùng giữa các phương pháp")
        render_intersection_section(results_by_method)

    render_evaluation_section()


if __name__ == "__main__":
    main()
