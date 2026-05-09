# Vietnamese Legal Document Retrieval

## 1. Giới thiệu

Dự án xây dựng hệ thống truy xuất văn bản pháp luật tiếng Việt sử dụng các kỹ thuật Information Retrieval cơ bản.

Hệ thống hỗ trợ so sánh 3 thuật toán:

1. Boolean Retrieval
2. TF-IDF + Cosine Similarity
3. BM25

Mục tiêu của project:

* Hiểu cơ chế hoạt động của hệ thống truy xuất thông tin.
* Tự cài đặt inverted index.
* Tự cài đặt TF-IDF và BM25.
* Đánh giá bằng các metric IR phổ biến.
* Áp dụng trên dữ liệu pháp luật tiếng Việt.

---

## 2. Dataset sử dụng

### Dataset chính

Vietnamese Legal Documents Dataset:

[https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents](https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents)

### Raw data

[https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval](https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval)

---

## 3. Cấu trúc dữ liệu

Dataset hiện gồm các split:

```text
train
test
```

Các cột chính:

| Column       | Ý nghĩa                      |
| ------------ | ---------------------------- |
| qid          | Query ID                     |
| question     | Câu hỏi pháp luật tiếng Việt |
| cid          | Danh sách document liên quan |
| context_list | Nội dung văn bản liên quan   |

---

## 4. Ánh xạ sang bài toán IR

| Thành phần IR | Dataset      |
| ------------- | ------------ |
| Query         | question     |
| Query ID      | qid          |
| Document      | context_list |
| Document ID   | cid          |
| Ground Truth  | qid -> cid   |

Sau khi chuẩn hóa dữ liệu, project sẽ tạo:

```text
data/processed/corpus.json
data/processed/queries.json
data/processed/qrels.json
```

---

## 5. Công nghệ sử dụng

| Thành phần            | Công nghệ             |
| --------------------- | --------------------- |
| Ngôn ngữ              | Python                |
| Dataset Loader        | Hugging Face datasets |
| Tiền xử lý tiếng Việt | underthesea           |
| Tính toán             | numpy, pandas         |
| Evaluation            | numpy                 |

---

## 6. Cấu trúc thư mục project

```text
vietnamese-legal-ir/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── corpus.json
│   │   ├── queries.json
│   │   └── qrels.json
│   │
│   └── sample/
│
├── outputs/
│   ├── indexes/
│   ├── results/
│   └── reports/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── indexing/
│   ├── retrieval/
│   ├── evaluation/
│   └── app/
│
├── notebooks/
└── tests/
```

---

## 7. Tạo repository GitHub

### Tạo repo

Tạo repository mới trên GitHub:

```text
vietnamese-legal-ir
```

Clone repo:

```bash
git clone https://github.com/USERNAME/vietnamese-legal-ir.git
cd vietnamese-legal-ir
```

---

## 8. Tạo môi trường Python

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 9. Cài thư viện

### Cài requirements

```bash
python -m pip install -r requirements.txt
```

### requirements.txt

```text
datasets
pandas
numpy
scikit-learn
underthesea
pyyaml
tqdm
pytest
```

Lưu ý:

Trên Windows nên luôn dùng:

```bash
python -m pip install ...
```

không nên dùng trực tiếp:

```bash
pip install ...
```

để tránh lỗi lệch môi trường virtual environment.

---

## 10. Kiểm tra môi trường

Kiểm tra Python đang dùng:

```powershell
python -c "import sys; print(sys.executable)"
```

Kiểm tra datasets:

```powershell
python -c "import datasets; print('datasets OK')"
```

Nếu thành công:

```text
datasets OK
```

---

## 11. Load dataset từ Hugging Face

Chạy:

```bash
python src/data/load_dataset.py
```

Ví dụ code:

```python
from datasets import load_dataset


ds = load_dataset("YuITC/Vietnamese-Legal-Documents")

print(ds)
print(ds["train"][0])
```

Kết quả mong đợi:

```text
train
test
```

và dữ liệu mẫu của dataset.

---

## 12. Chuẩn hóa dữ liệu

Chạy script:

```bash
python src/data/prepare_data.py --max_queries 100 --max_docs 1000
```

Kết quả:

```text
data/processed/corpus.json
data/processed/queries.json
data/processed/qrels.json
```

Ý nghĩa:

| File         | Vai trò                         |
| ------------ | ------------------------------- |
| corpus.json  | Toàn bộ văn bản pháp luật       |
| queries.json | Danh sách câu hỏi               |
| qrels.json   | Ground truth document liên quan |

---

## 13. Tiền xử lý văn bản

Các bước preprocessing:

1. Lowercase.
2. Loại bỏ ký tự đặc biệt.
3. Chuẩn hóa khoảng trắng.
4. Tách từ tiếng Việt.
5. Loại bỏ stopwords.

Ví dụ:

```text
Người lao động được nghỉ hằng năm bao nhiêu ngày?
```

Sau tokenize:

```text
['người', 'lao_động', 'nghỉ', 'hằng_năm', 'bao_nhiêu', 'ngày']
```

---

## 14. Boolean Retrieval

### Người phụ trách

Người 1

### Thành phần chính

```text
src/indexing/inverted_index.py
src/retrieval/boolean_retrieval.py
```

### Chức năng

* Xây dựng inverted index.
* Tạo dictionary và postings list.
* Hỗ trợ AND.
* Hỗ trợ OR.
* Hỗ trợ NOT nếu kịp.

### Build inverted index

```bash
python src/indexing/inverted_index.py
```

Kết quả:

```text
outputs/indexes/inverted_index.json
```

### Chạy Boolean Retrieval

```bash
python src/app/cli.py --method boolean --query "điều kiện kết hôn" --top_k 5
```

### Nhận xét

Ưu điểm:

* Dễ cài đặt.
* Tìm kiếm nhanh với inverted index.
* Minh họa rõ cơ chế search engine.

Hạn chế:

* Không ranking tốt.
* Phụ thuộc từ khóa chính xác.

---

## 15. TF-IDF + Cosine Similarity

### Người phụ trách

Người 2

### Thành phần chính

```text
src/retrieval/tfidf_retrieval.py
```

### Công thức

```text
tfidf(t,d) = tf(t,d) * idf(t)
```

```text
cosine(q,d) = (q · d) / (||q|| * ||d||)
```

### Chạy TF-IDF

```bash
python src/app/cli.py --method tfidf --query "điều kiện kết hôn" --top_k 5
```

### Nhận xét

Ưu điểm:

* Có ranking.
* Dễ hiểu.
* Phù hợp baseline.

Hạn chế:

* Nhạy với cách tokenize.
* Chưa xử lý tốt độ dài văn bản.

---

## 16. BM25 Retrieval

### Người phụ trách

Người 3

### Thành phần chính

```text
src/retrieval/bm25_retrieval.py
```

### Công thức

```text
score(q,d) = Σ IDF(t) * ((tf(t,d)*(k1+1)) /
              (tf(t,d) + k1*(1-b+b*dl/avgdl)))
```

Tham số mặc định:

```text
k1 = 1.5
b = 0.75
```

### Chạy BM25

```bash
python src/app/cli.py --method bm25 --query "điều kiện kết hôn" --top_k 5
```

### Nhận xét

Ưu điểm:

* Ranking tốt hơn TF-IDF.
* Phù hợp search engine thực tế.

Hạn chế:

* Tính toán phức tạp hơn Boolean.

---

## 17. Đánh giá hệ thống

### Metrics sử dụng

```text
Precision@K
Recall@K
MAP
nDCG@K
```

### Chạy evaluation

```bash
python src/evaluation/evaluate.py --method all --top_k 10 --max_queries 100
```

Kết quả lưu tại:

```text
outputs/reports/evaluation_report.csv
```

---

## 18. Ví dụ CLI

### Boolean

```bash
python src/app/cli.py --method boolean --query "điều kiện kết hôn" --top_k 5
```

### TF-IDF

```bash
python src/app/cli.py --method tfidf --query "điều kiện kết hôn" --top_k 5
```

### BM25

```bash
python src/app/cli.py --method bm25 --query "điều kiện kết hôn" --top_k 5
```

---

## 19. Unit Tests

Chạy test:

```bash
pytest tests/
```

Các test gồm:

```text
test_preprocessing.py
test_boolean.py
test_tfidf.py
test_bm25.py
test_metrics.py
```

---

## 20. Luồng chạy end-to-end

### Bước 1

```bash
python src/data/prepare_data.py --max_queries 100 --max_docs 2000
```

### Bước 2

```bash
python src/indexing/inverted_index.py
```

### Bước 3

```bash
python src/app/cli.py --method boolean --query "điều kiện kết hôn" --top_k 5
```

### Bước 4

```bash
python src/app/cli.py --method tfidf --query "điều kiện kết hôn" --top_k 5
```

### Bước 5

```bash
python src/app/cli.py --method bm25 --query "điều kiện kết hôn" --top_k 5
```

### Bước 6

```bash
python src/evaluation/evaluate.py --method all --top_k 10 --max_queries 100
```

---

## 21. Hướng phát triển

Có thể mở rộng project theo các hướng:

1. Query expansion.
2. Sửa lỗi chính tả tiếng Việt.
3. Phrase search.
4. Dense Retrieval.
5. Streamlit UI.
6. RAG chatbot pháp luật.

---

## 22. Kết luận

Project giúp hiểu rõ các thành phần chính của Information Retrieval:

* Tiền xử lý văn bản.
* Inverted Index.
* Boolean Retrieval.
* TF-IDF.
* BM25.
* Evaluation.

Trong ba thuật toán:

* Boolean Retrieval phù hợp minh họa inverted index.
* TF-IDF là baseline dễ hiểu.
* BM25 thường cho kết quả tốt nhất.

---

## 23. Tài liệu tham khảo

1. Introduction to Information Retrieval - Manning, Raghavan, Schütze.
2. Hugging Face Datasets Documentation.
3. Vietnamese Legal Documents Dataset.
4. BM25 Okapi Retrieval.
