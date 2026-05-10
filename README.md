# Vietnamese Legal Document Retrieval

Dự án xây dựng hệ thống truy xuất văn bản pháp luật tiếng Việt bằng các kỹ thuật **Information Retrieval** cơ bản. Project tập trung vào việc tự cài đặt các thành phần chính thay vì dùng search engine có sẵn như Apache Lucene.

## 1. Mục tiêu

Project hướng tới các mục tiêu:

- Chuẩn hóa dữ liệu pháp luật tiếng Việt thành `corpus`, `queries`, `qrels`.
- Tiền xử lý văn bản tiếng Việt bằng chuẩn hóa text, tokenizer và stopwords.
- Xây dựng **Inverted Index**.
- Cài đặt và so sánh 3 phương pháp truy xuất:
  - Boolean Retrieval
  - TF-IDF + Cosine Similarity
  - BM25
- Đánh giá kết quả bằng các metric IR: Precision@K, Recall@K, MAP, nDCG@K.

## 2. Dataset sử dụng

### Dataset chính

- Vietnamese Legal Documents Dataset
- Link: <https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents>
- Tên repo khi load bằng Hugging Face:

```python
YuITC/Vietnamese-Legal-Documents
```

### Raw data

- BKAI Legal Retrieval
- Link: <https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval>

## 3. Cấu trúc dữ liệu

Dataset gồm 2 split chính:

```text
train
test
```

Các cột chính:

| Cột | Ý nghĩa |
|---|---|
| `qid` | ID của câu hỏi / query |
| `question` | Câu hỏi pháp luật tiếng Việt |
| `cid` | Danh sách ID văn bản liên quan |
| `context_list` | Danh sách nội dung văn bản liên quan |

Sau khi chuẩn hóa, dữ liệu được chuyển sang dạng phù hợp với bài toán IR:

| Thành phần IR | File sinh ra | Ý nghĩa |
|---|---|---|
| Corpus | `data/processed/corpus.json` | Tập văn bản pháp luật, ánh xạ `doc_id -> text` |
| Queries | `data/processed/queries.json` | Tập câu hỏi, ánh xạ `qid -> question` |
| Qrels | `data/processed/qrels.json` | Ground truth, ánh xạ `qid -> list[doc_id]` |

## 4. Cấu trúc thư mục

```text
vietnamese-legal-ir/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── corpus.json
│   │   ├── queries.json
│   │   └── qrels.json
│   └── sample/
├── outputs/
│   ├── indexes/
│   │   └── inverted_index.json
│   ├── results/
│   └── reports/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── evaluation/
│   └── app/
├── notebooks/
└── tests/
```

## 5. Cài đặt môi trường

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Kiểm tra Python đang dùng đúng môi trường ảo:

```powershell
python -c "import sys; print(sys.executable)"
```

Kiểm tra thư viện `datasets`:

```powershell
python -c "import datasets; print('datasets OK')"
```

## 6. Lưu ý quan trọng khi chạy lệnh

Project sử dụng package `src`, vì vậy nên chạy các script bằng dạng module:

```powershell
python -m src.package.module
```

Không nên chạy trực tiếp kiểu:

```powershell
python src/package/module.py
```

Ví dụ đúng:

```powershell
python -m src.indexing.inverted_index
```

Ví dụ dễ gây lỗi `ModuleNotFoundError: No module named 'src'`:

```powershell
python src/indexing/inverted_index.py
```

## 7. Chuẩn bị dữ liệu

Chạy lệnh sau từ thư mục root của project:

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
```

Kết quả sinh ra:

```text
data/processed/corpus.json
data/processed/queries.json
data/processed/qrels.json
```

Có thể chạy với dữ liệu nhỏ hơn để test nhanh:

```powershell
python -m src.data.prepare_data --max_queries 100 --max_docs 1000
```

## 8. Tiền xử lý văn bản

Pipeline preprocessing gồm:

1. Chuẩn hóa Unicode nếu có.
2. Chuyển chữ thường.
3. Loại bỏ ký tự đặc biệt không cần thiết.
4. Chuẩn hóa khoảng trắng.
5. Tách từ tiếng Việt bằng `underthesea`.
6. Loại bỏ stopwords tiếng Việt.

Ví dụ input:

```text
Người lao động được nghỉ hằng năm bao nhiêu ngày?
```

Ví dụ token sau xử lý:

```text
['người', 'lao_động', 'nghỉ', 'hằng_năm', 'ngày']
```

Các file chính:

```text
src/preprocessing/text_cleaner.py
src/preprocessing/tokenizer.py
src/preprocessing/stopwords.py
```

## 9. Phân công nhóm

| Thành viên | Phụ trách | File chính |
|---|---|---|
| Người 1 | Boolean Retrieval + Inverted Index | `src/indexing/inverted_index.py`, `src/retrieval/boolean_retrieval.py` |
| Người 2 | TF-IDF + Cosine Similarity | `src/retrieval/tfidf_retrieval.py` |
| Người 3 | BM25 + Evaluation | `src/retrieval/bm25_retrieval.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluate.py` |

## 10. Boolean Retrieval

### Trạng thái

Phần Boolean Retrieval và Inverted Index đã có thể chạy end-to-end với dữ liệu mẫu từ dataset.

### Chức năng

- Xây dựng inverted index.
- Tạo postings list dạng `term -> list[doc_id]`.
- Lưu vocabulary, document frequency và document length.
- Hỗ trợ truy vấn tự nhiên, mặc định dùng AND giữa các token.
- Hỗ trợ toán tử `OR`.
- Hỗ trợ toán tử `NOT`.
- Trả về Top-K document phù hợp.

### Build inverted index

Sau khi đã chuẩn bị dữ liệu, chạy:

```powershell
python -m src.indexing.inverted_index
```

Kết quả lưu tại:

```text
outputs/indexes/inverted_index.json
```

Ví dụ output:

```text
Đã build inverted index.
Số document: 558
Kích thước vocabulary: 4632
Lưu tại: outputs/indexes/inverted_index.json
```

### Chạy Boolean Retrieval trực tiếp

```powershell
python -m src.retrieval.boolean_retrieval --query "điều kiện kết hôn" --top_k 5
```

Ví dụ query có `OR`:

```powershell
python -m src.retrieval.boolean_retrieval --query "kết hôn OR lao động" --top_k 5
```

Ví dụ query có `NOT`:

```powershell
python -m src.retrieval.boolean_retrieval --query "kết hôn NOT ly hôn" --top_k 5
```

### Chạy qua CLI

```powershell
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
```

Ví dụ output:

```text
Query: điều kiện kết hôn
Method: boolean

1. CID: 47569
   Score: 2.0
   Text: "Điều 8. Điều kiện kết hôn ..."
```

### Nhận xét

Ưu điểm:

- Dễ cài đặt.
- Tốc độ nhanh sau khi đã có inverted index.
- Minh họa rõ cơ chế tìm kiếm bằng postings list.

Hạn chế:

- Phụ thuộc nhiều vào từ khóa chính xác.
- Không hiểu ngữ nghĩa.
- Khả năng ranking kém hơn TF-IDF và BM25.

## 11. TF-IDF + Cosine Similarity

### Người phụ trách

Người 2.

### Mục tiêu

- Xây dựng vocabulary.
- Tính TF, IDF, TF-IDF.
- Biểu diễn query và document thành vector.
- Tính cosine similarity.
- Trả về Top-K document có score cao nhất.

### Công thức

```text
tfidf(t, d) = tf(t, d) * idf(t)
```

```text
cosine(q, d) = (q · d) / (||q|| * ||d||)
```

### Lệnh chạy dự kiến

```powershell
python -m src.app.cli --method tfidf --query "điều kiện kết hôn" --top_k 5
```

## 12. BM25 Retrieval

### Người phụ trách

Người 3.

### Mục tiêu

- Tính term frequency trong document.
- Tính document length và average document length.
- Cài đặt BM25 Okapi.
- Trả về Top-K document theo điểm BM25.

### Công thức

```text
score(q, d) = Σ IDF(t) * ((tf(t,d) * (k1 + 1)) /
              (tf(t,d) + k1 * (1 - b + b * dl / avgdl)))
```

Tham số mặc định:

```text
k1 = 1.5
b = 0.75
```

### Lệnh chạy dự kiến

```powershell
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
```

## 13. Evaluation

### Metrics

Các metric sử dụng:

- Precision@K
- Recall@K
- MAP
- nDCG@K

### Lệnh chạy dự kiến

```powershell
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
```

Kết quả lưu tại:

```text
outputs/reports/evaluation_report.csv
```

## 14. Unit tests

Chạy toàn bộ test:

```powershell
python -m pytest
```

Chạy riêng test Boolean:

```powershell
python -m pytest tests/test_boolean.py -v
```

Kết quả mong đợi hiện tại cho phần Boolean:

```text
5 passed
```

Các nhóm test nên có:

```text
tests/test_preprocessing.py
tests/test_boolean.py
tests/test_tfidf.py
tests/test_bm25.py
tests/test_metrics.py
```

## 15. Luồng chạy end-to-end hiện tại cho Boolean Retrieval

Chạy lần lượt:

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
python -m src.indexing.inverted_index
python -m src.retrieval.boolean_retrieval --query "điều kiện kết hôn" --top_k 5
python -m pytest
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
```

Nếu chạy thành công, phần Người 1 đã hoàn thành pipeline Boolean Retrieval.

## 16. Luồng chạy end-to-end sau khi hoàn thành cả 3 thuật toán

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
python -m src.indexing.inverted_index
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
python -m src.app.cli --method tfidf --query "điều kiện kết hôn" --top_k 5
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
```

## 17. Troubleshooting

### Lỗi `ModuleNotFoundError: No module named 'src'`

Nguyên nhân thường gặp là chạy script trực tiếp bằng đường dẫn file.

Không nên chạy:

```powershell
python src/indexing/inverted_index.py
```

Hãy chạy:

```powershell
python -m src.indexing.inverted_index
```

### Lỗi chưa có `corpus.json`

Hãy chạy prepare data trước:

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
```

### Lỗi chưa có `inverted_index.json`

Hãy build index trước:

```powershell
python -m src.indexing.inverted_index
```

### Query Boolean không ra kết quả

Một số nguyên nhân có thể là:

- Token trong query không khớp với token sau preprocessing.
- Query quá cụ thể hoặc dùng từ không xuất hiện trong corpus nhỏ.
- Dữ liệu đang chạy chỉ là subset do dùng `--max_queries`, `--max_docs`.

Có thể thử query đơn giản hơn:

```powershell
python -m src.app.cli --method boolean --query "kết hôn" --top_k 5
```

hoặc:

```powershell
python -m src.app.cli --method boolean --query "lao động" --top_k 5
```

## 18. Hướng phát triển

Có thể mở rộng project theo các hướng:

1. Hoàn thiện TF-IDF retrieval.
2. Hoàn thiện BM25 retrieval.
3. Tối ưu scoring bằng candidate documents từ inverted index.
4. Bổ sung phrase search.
5. Bổ sung query expansion.
6. Bổ sung sửa lỗi chính tả tiếng Việt.
7. Xây dựng giao diện Streamlit.
8. Dùng kết quả truy xuất làm ngữ cảnh cho chatbot RAG pháp luật.

## 19. Kết luận

Project giúp hiểu rõ các thành phần chính của hệ thống Information Retrieval:

- Tiền xử lý văn bản.
- Inverted Index.
- Boolean Retrieval.
- TF-IDF.
- BM25.
- Evaluation.

Trong ba thuật toán:

- Boolean Retrieval phù hợp để minh họa cơ chế inverted index và truy vấn logic.
- TF-IDF là baseline có ranking, dễ hiểu và dễ giải thích.
- BM25 thường hiệu quả hơn TF-IDF trong tìm kiếm văn bản thực tế.

## 20. Tài liệu tham khảo

1. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2009.
2. Hugging Face Datasets Documentation: <https://huggingface.co/docs/datasets>
3. Vietnamese Legal Documents Dataset: <https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents>
4. BKAI Legal Retrieval Raw Dataset: <https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval>
5. BM25 Okapi Retrieval.

## 21. Demo giao diện Streamlit

Mục đích demo:

- nhập query pháp luật tiếng Việt
- chạy đồng thời Boolean Retrieval, TF-IDF và BM25
- so sánh ranking, score và thời gian chạy trên cùng một màn hình
- hiển thị metric evaluation nếu đã có file báo cáo

Chuẩn bị full data:

```powershell
python -m src.data.prepare_data
python -m src.indexing.inverted_index
```

Chạy giao diện demo:

```powershell
streamlit run src/app/demo_streamlit.py
```

Giao diện demo gồm:

- sidebar kiểm tra trạng thái `corpus.json`, `queries.json`, `qrels.json`, `inverted_index.json`
- ô nhập query và các query mẫu để thao tác nhanh khi bảo vệ
- bảng tổng quan so sánh 3 phương pháp theo số kết quả, thời gian tìm kiếm và best score
- 3 tab kết quả riêng cho Boolean Retrieval, TF-IDF và BM25
- bảng đối chiếu tài liệu theo rank và phần giao nhau giữa các phương pháp
- khu vực hiển thị `evaluation_report.csv` và `summary.md` nếu đã chạy đánh giá

## 22. Tối ưu full data và chạy demo nhanh

Quy trình lần đầu:

```powershell
python -m src.data.prepare_data
python -m src.indexing.build_retrieval_models
streamlit run src/app/demo_streamlit.py
```

Quy trình các lần sau:

```powershell
streamlit run src/app/demo_streamlit.py
```

Giải thích:

- `build_retrieval_models.py` sẽ kiểm tra `corpus.json`, `queries.json`, `qrels.json`, `inverted_index.pkl`, `tfidf_model.pkl`, `bm25_model.pkl`.
- Nếu artifact đã tồn tại thì script sẽ `SKIP` và không build lại.
- Streamlit chỉ load artifact đã lưu, không fit lại TF-IDF hoặc BM25 khi mở app.
- Muốn build lại từ đầu thì chạy:

```powershell
python -m src.indexing.build_retrieval_models --force
```
