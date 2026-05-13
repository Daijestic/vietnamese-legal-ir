# Vietnamese Legal Document Retrieval

![Python](https://img.shields.io/badge/python-3.11-blue)
![IR](https://img.shields.io/badge/project-information_retrieval-green)
![Pure Python](https://img.shields.io/badge/implementation-pure_python-orange)

Dự án xây dựng hệ thống truy xuất văn bản pháp luật tiếng Việt bằng các kỹ thuật **Information Retrieval** cơ bản.
Project tập trung vào việc **tự cài đặt thuật toán bằng Python thuần** thay vì dùng search engine hoặc thư viện IR có sẵn như Apache Lucene, Elasticsearch, `rank_bm25` hay `scikit-learn` TF-IDF Vectorizer.

---

# 1. Mục tiêu

Project hướng tới các mục tiêu:

* Chuẩn hóa dữ liệu pháp luật tiếng Việt thành `corpus`, `queries`, `qrels`
* Tiền xử lý văn bản tiếng Việt
* Xây dựng **Inverted Index**
* Cài đặt và so sánh 3 phương pháp truy xuất:

  * Boolean Retrieval
  * TF-IDF + Cosine Similarity
  * BM25
* Đánh giá bằng các metric:

  * Precision@K
  * Recall@K
  * MAP
  * nDCG@K

---

# 2. Highlights

* Pure Python Information Retrieval implementation
* Handwritten TF-IDF and BM25
* Handwritten Inverted Index
* Handwritten Cosine Similarity
* Handwritten Evaluation Metrics
* Vietnamese Legal Document Retrieval
* Streamlit Demo UI
* End-to-end Retrieval Pipeline

---

# 3. Pure Python Implementation

Project được xây dựng theo hướng:

```text
Code tay tối đa
```

để phục vụ mục tiêu học thuật của môn Information Retrieval.

## Các thành phần được tự cài đặt bằng Python thuần

* Inverted Index
* Boolean Retrieval
* TF-IDF
* Cosine Similarity
* BM25
* Precision@K
* Recall@K
* MAP
* nDCG

## Project KHÔNG sử dụng

* `scikit-learn` TF-IDF Vectorizer
* `rank_bm25`
* thư viện search engine
* thư viện retrieval
* thư viện evaluation IR
* tokenizer NLP ngoài

## Các thư viện còn sử dụng chỉ phục vụ

| Library     | Mục đích      |
| ----------- | ------------- |
| `datasets`  | Load dataset  |
| `tqdm`      | Progress bar  |
| `streamlit` | Demo UI       |
| `pandas`    | Hiển thị bảng |
| `pytest`    | Testing       |

---

# 4. Dataset sử dụng

## Dataset chính

Vietnamese Legal Documents Dataset

Link:

```text
https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents
```

Tên repo khi load bằng Hugging Face:

```python
YuITC/Vietnamese-Legal-Documents
```

## Raw Dataset

BKAI Legal Retrieval Dataset

```text
https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval
```

---

# 5. Cấu trúc dữ liệu

Dataset gồm:

```text
train
test
```

## Các cột chính

| Cột            | Ý nghĩa                      |
| -------------- | ---------------------------- |
| `qid`          | Query ID                     |
| `question`     | Câu hỏi pháp luật            |
| `cid`          | Danh sách document liên quan |
| `context_list` | Nội dung văn bản pháp luật   |

---

# 6. Chuẩn hóa sang bài toán IR

Sau khi chuẩn hóa:

| Thành phần | File           | Ý nghĩa                   |
| ---------- | -------------- | ------------------------- |
| Corpus     | `corpus.json`  | `doc_id -> text`          |
| Queries    | `queries.json` | `qid -> question`         |
| Qrels      | `qrels.json`   | `qid -> relevant_doc_ids` |

Các file được lưu tại:

```text
data/processed/
```

---

# 7. Cấu trúc thư mục project

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
│   ├── reports/
│   └── results/
├── src/
│   ├── app/
│   ├── data/
│   ├── evaluation/
│   ├── indexing/
│   ├── preprocessing/
│   └── retrieval/
├── notebooks/
└── tests/
```

---

# 8. Cài đặt môi trường

## Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

# 9. Kiểm tra môi trường

## Kiểm tra Python đang dùng

```powershell
python -c "import sys; print(sys.executable)"
```

## Kiểm tra datasets

```powershell
python -c "import datasets; print('datasets OK')"
```

Nếu đúng:

```text
datasets OK
```

---

# 10. Lưu ý quan trọng khi chạy project

Project sử dụng package `src`.

Vì vậy nên chạy bằng:

```powershell
python -m src.package.module
```

Ví dụ đúng:

```powershell
python -m src.indexing.inverted_index
```

KHÔNG nên chạy:

```powershell
python src/indexing/inverted_index.py
```

vì dễ gây lỗi:

```text
ModuleNotFoundError: No module named 'src'
```

---

# 11. Chuẩn bị dữ liệu

## Chạy prepare_data

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
```

Kết quả:

```text
data/processed/corpus.json
data/processed/queries.json
data/processed/qrels.json
```

## Test nhanh với subset nhỏ

```powershell
python -m src.data.prepare_data --max_queries 100 --max_docs 1000
```

---

# 12. Tiền xử lý văn bản

Pipeline preprocessing:

1. Lowercase
2. Regex cleaning
3. Normalize whitespace
4. Split whitespace
5. Remove stopwords

Tokenizer hiện tại sử dụng Python thuần:

```python
text.split()
```

Project cố ý KHÔNG dùng:

* underthesea
* pyvi
* tokenizer NLP

để đảm bảo đúng yêu cầu:

```text
tự cài đặt thuật toán
```

## Ví dụ

Input:

```text
Người lao động được nghỉ hằng năm bao nhiêu ngày?
```

Output:

```python
['người', 'lao', 'động', 'nghỉ', 'hằng', 'năm', 'bao', 'nhiêu', 'ngày']
```

## File chính

```text
src/preprocessing/text_cleaner.py
src/preprocessing/tokenizer.py
src/preprocessing/stopwords.py
```

---

# 13. Phân công nhóm

| Thành viên | Phụ trách                                          |
| ---------- | -------------------------------------------------- |
| Người 1    | Preprocessing + Inverted Index + Boolean Retrieval |
| Người 2    | TF-IDF + Cosine Similarity                         |
| Người 3    | BM25 + Evaluation                                  |

---

# 14. Boolean Retrieval

## Thành phần chính

```text
src/indexing/inverted_index.py
src/retrieval/boolean_retrieval.py
```

## Chức năng

* Xây inverted index
* Tạo postings list
* Hỗ trợ:

  * AND
  * OR
  * NOT
* Trả về Top-K document

## Build inverted index

```powershell
python -m src.indexing.inverted_index
```

Kết quả:

```text
outputs/indexes/inverted_index.json
```

## Chạy Boolean Retrieval

```powershell
python -m src.retrieval.boolean_retrieval --query "điều kiện kết hôn" --top_k 5
```

Hoặc:

```powershell
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
```

---

# 15. TF-IDF + Cosine Similarity

## Thành phần chính

```text
src/retrieval/tfidf_retrieval.py
```

## Tự cài đặt

* TF
* IDF
* TF-IDF vector
* Query vector
* Cosine similarity

## Công thức

```text
tfidf(t,d) = tf(t,d) * idf(t)
```

```text
cosine(q,d) = (q · d) / (||q|| * ||d||)
```

## Chạy TF-IDF

```powershell
python -m src.app.cli --method tfidf --query "điều kiện kết hôn" --top_k 5
```

---

# 16. BM25 Retrieval

## Thành phần chính

```text
src/retrieval/bm25_retrieval.py
```

## Tự cài đặt

* tf
* df
* avgdl
* BM25 score
* document length normalization

## Công thức

```text
score(q,d) =
Σ IDF(t) *
((tf(t,d)*(k1+1)) /
(tf(t,d)+k1*(1-b+b*dl/avgdl)))
```

## Tham số

```text
k1 = 1.5
b = 0.75
```

## Chạy BM25

```powershell
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
```

---

# 17. Evaluation

## Metrics

* Precision@K
* Recall@K
* MAP
* nDCG@K

Toàn bộ metric được tự cài đặt bằng Python thuần.

## Chạy evaluation

```powershell
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
```

Kết quả:

```text
outputs/reports/evaluation_report.csv
```

---

# 18. Unit Tests

## Chạy toàn bộ test

```powershell
python -m pytest tests -q
```

## Kết quả hiện tại

```text
20 passed
```

## Các test chính

```text
tests/test_preprocessing.py
tests/test_boolean.py
tests/test_tfidf.py
tests/test_bm25.py
tests/test_metrics.py
```

---

# 19. Luồng chạy end-to-end

## Bước 1 — Prepare Data

```powershell
python -m src.data.prepare_data --max_queries 500 --max_docs 5000
```

## Bước 2 — Build Index

```powershell
python -m src.indexing.inverted_index
```

## Bước 3 — Boolean Retrieval

```powershell
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
```

## Bước 4 — TF-IDF

```powershell
python -m src.app.cli --method tfidf --query "điều kiện kết hôn" --top_k 5
```

## Bước 5 — BM25

```powershell
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
```

## Bước 6 — Evaluation

```powershell
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
```

---

# 20. Streamlit Demo

## Chạy demo

```powershell
streamlit run src/app/demo_streamlit.py
```

## Chức năng demo

* nhập query pháp luật tiếng Việt
* chạy Boolean / TF-IDF / BM25
* so sánh ranking và score
* hiển thị evaluation report
* so sánh thời gian chạy

---

# 21. Kết quả kiểm thử

Project đã được kiểm thử thành công với:

```powershell
python -m pytest tests -q
```

Kết quả:

```text
20 passed
```

Các pipeline đã xác minh chạy thành công:

```powershell
python -m src.data.prepare_data
python -m src.indexing.inverted_index
python -m src.app.cli --method boolean --query "điều kiện kết hôn"
python -m src.app.cli --method tfidf --query "điều kiện kết hôn"
python -m src.app.cli --method bm25 --query "điều kiện kết hôn"
python -m src.evaluation.evaluate
```

---

# 22. Troubleshooting

## ModuleNotFoundError: No module named 'src'

KHÔNG chạy:

```powershell
python src/indexing/inverted_index.py
```

Hãy chạy:

```powershell
python -m src.indexing.inverted_index
```

---

## Chưa có corpus.json

Chạy:

```powershell
python -m src.data.prepare_data
```

---

## Chưa có inverted_index.json

Chạy:

```powershell
python -m src.indexing.inverted_index
```

---

## Boolean query không ra kết quả

Thử query đơn giản hơn:

```powershell
python -m src.app.cli --method boolean --query "kết hôn"
```

hoặc:

```powershell
python -m src.app.cli --method boolean --query "lao động"
```

---

# 23. Hướng phát triển

Có thể mở rộng project theo các hướng:

1. Phrase Search
2. Query Expansion
3. Vietnamese Spell Correction
4. Dense Retrieval
5. RAG Chatbot
6. Hybrid Search
7. Better Vietnamese Tokenization
8. Streamlit Deployment

---

# 24. Kết luận

Project giúp hiểu rõ các thành phần cốt lõi của hệ thống Information Retrieval:

* Preprocessing
* Inverted Index
* Boolean Retrieval
* TF-IDF
* BM25
* Evaluation

Trong ba phương pháp:

* Boolean Retrieval minh họa rõ cơ chế postings list
* TF-IDF là baseline dễ hiểu
* BM25 thường cho kết quả retrieval tốt nhất

Project được xây dựng theo hướng:

```text
Pure Python IR Implementation
```

để phục vụ mục tiêu học thuật và giúp hiểu sâu bản chất thuật toán.

---

# 25. Tài liệu tham khảo

1. Christopher D. Manning, Prabhakar Raghavan, Hinrich Schütze — *Introduction to Information Retrieval*
2. Hugging Face Datasets Documentation
3. Vietnamese Legal Documents Dataset
4. BM25 Okapi Retrieval
5. Information Retrieval Lecture Notes
