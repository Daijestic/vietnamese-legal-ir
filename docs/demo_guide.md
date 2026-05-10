# Demo Guide

## Chuẩn bị dữ liệu

Chạy full data từ thư mục root của project:

```bash
python -m src.data.prepare_data
python -m src.indexing.build_retrieval_models
```

Neu muon tao bang metric truoc khi demo:

```bash
python -m src.evaluation.evaluate --method all --top_k 10
python -m src.evaluation.generate_summary
```

## Chạy demo

```bash
streamlit run src/app/demo_streamlit.py
```

Giao dien se:

- Nhap query phap luat tieng Viet.
- Chay dong thoi Boolean Retrieval, TF-IDF va BM25.
- Hien thi ranking, score, search time va cac tai lieu trung nhau.
- Hien thi metric evaluation neu da co `outputs/reports/evaluation_report.csv`.

## Query mẫu khi bảo vệ

- điều kiện kết hôn theo pháp luật Việt Nam
- người lao động được nghỉ hằng năm bao nhiêu ngày
- hợp đồng lao động bị chấm dứt trong trường hợp nào
- quyền nuôi con sau khi ly hôn
- xử phạt vi phạm giao thông đường bộ

## Cách giải thích kết quả

- Boolean Retrieval phụ thuộc mạnh vào khớp từ khóa nên thường chính xác với truy vấn rõ từ khóa, nhưng ít linh hoạt.
- TF-IDF có ranking theo trọng số từ, giúp tài liệu chứa từ quan trọng nổi lên cao hơn.
- BM25 thường tốt hơn trên văn bản dài vì có cơ chế chuẩn hóa độ dài tài liệu và term frequency ổn định hơn.
