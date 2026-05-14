# Vietnamese Legal IR

Project truy xuất văn bản pháp luật tiếng Việt bằng Python thuần với 3 phương pháp:

- Boolean Retrieval
- TF-IDF + cosine similarity
- BM25

Project giữ hướng cài đặt thuật toán bằng tay, không dùng thư viện IR/search engine có sẵn.

## Dataset sử dụng

### 1. `tmnam20/BKAI-Legal-Retrieval`

Link: <https://huggingface.co/datasets/tmnam20/BKAI-Legal-Retrieval>

Dataset này dùng để tạo:

- `corpus.json`
- inverted index
- TF-IDF model
- BM25 model

Cột được dùng để build corpus:

- `cid`
- `text`

Schema thực tế của snapshot raw hiện tại:

- `corpus.csv`: `text`, `cid`
- `public_test.csv`: `question`, `qid`
- `train.csv`: `question`, `context`, `cid`, `qid`
- `train_split.csv`: `question`, `context`, `cid`, `qid`
- `val_split.csv`: `question`, `context`, `cid`, `qid`

Project chỉ lấy `corpus.csv` để build tập tài liệu.

### 2. `YuITC/Vietnamese-Legal-Documents`

Link: <https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents>

Dataset này chỉ dùng cho evaluation.

Schema thực tế:

- `train`: `question`, `context_list`, `qid`, `cid`
- `test`: `question`, `context_list`, `qid`, `cid`

Project dùng:

- `question` làm query
- `qid` làm query id
- `cid` làm ground-truth relevant documents

Lưu ý:

- mọi `qid` và `cid` đều được ép về `str`
- nếu thiếu `qid` thì script tự sinh theo index dòng
- `cid` có thể là list
- mỗi query có thể có từ 1 đến 9 relevant documents

## Pure Python constraints

Không dùng:

- `scikit-learn` `TfidfVectorizer`
- `sklearn.cosine_similarity`
- `rank_bm25`
- Lucene
- Elasticsearch
- thư viện IR/search engine tương tự

Được dùng:

- `datasets`
- `json`
- `math`
- `re`
- `collections`
- `pickle`
- `tqdm`

## Cài môi trường

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Lệnh chuẩn:

```powershell
python -m src.data.prepare_data --max_docs 5000 --max_queries 500
```

Tham số hỗ trợ:

- `--max_docs`
- `--max_queries`
- `--raw_dataset_name` mặc định `tmnam20/BKAI-Legal-Retrieval`
- `--eval_dataset_name` mặc định `YuITC/Vietnamese-Legal-Documents`

Output:

```text
data/processed/corpus.json
data/processed/queries.json
data/processed/qrels.json
```

Schema output:

```json
{
  "cid": "text"
}
```

```json
{
  "qid": "question"
}
```

```json
{
  "qid": ["cid1", "cid2"]
}
```

Khi `cid` trong qrels không tồn tại trong corpus raw:

- script không crash
- vẫn giữ nguyên qrels
- in warning
- báo số lượng missing `cid`
- báo tỷ lệ `cid` tìm thấy trong corpus

## Build index và retrieval models

Build inverted index:

```powershell
python -m src.indexing.inverted_index
```

Build/rebuild artifacts:

```powershell
python -m src.indexing.build_retrieval_models
```

Artifacts:

```text
outputs/indexes/inverted_index.json
outputs/indexes/inverted_index.pkl
outputs/indexes/tfidf_model.pkl
outputs/indexes/bm25_model.pkl
```

Các artifact này luôn được build từ `data/processed/corpus.json`, tức corpus lấy từ `tmnam20/BKAI-Legal-Retrieval`.

## Search CLI

CLI hỗ trợ:

- `--method {boolean,tfidf,bm25}`
- `--query`
- `--top_k <int|none>`
- `--threshold <float>`
- `--display_limit <int>` mặc định `20`

### Search top 5

```powershell
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
```

### Search không giới hạn top_k, dùng threshold

```powershell
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k none --threshold 1.5 --display_limit 20
```

Semantics của `search(query, top_k=10, threshold=0.0)`:

- nếu `top_k` là số nguyên: trả đúng Top-K
- nếu `top_k=None`: trả toàn bộ document có `score > threshold`
- Boolean: score = số token query match
- TF-IDF: score = cosine similarity
- BM25: score = BM25 score

## Evaluation

Evaluation dùng:

- `queries.json` và `qrels.json` từ `YuITC/Vietnamese-Legal-Documents`
- corpus/index/model build từ `tmnam20/BKAI-Legal-Retrieval`

Metrics hỗ trợ:

- `Precision`
- `Recall`
- `Average Precision`
- `Precision@5`
- `Recall@5`
- `Precision@10`
- `Recall@10`
- `nDCG@5`
- `nDCG@10`
- `MAP` (mean của average precision theo query)

Lưu ý:

- không giả định mỗi query chỉ có 1 relevant document
- `Recall = số relevant doc tìm được / tổng số relevant doc của query`

### Evaluate với top_k = 10

```powershell
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
```

### Evaluate với top_k = none

```powershell
python -m src.evaluation.evaluate --method all --top_k none --threshold 1.5 --max_queries 100
```

Output:

```text
outputs/reports/evaluation_report.csv
```

## Luồng chạy nhanh

```powershell
python -m src.data.prepare_data --max_docs 5000 --max_queries 500
python -m src.indexing.inverted_index
python -m src.indexing.build_retrieval_models
python -m src.app.cli --method boolean --query "điều kiện kết hôn" --top_k 5
python -m src.app.cli --method tfidf --query "điều kiện kết hôn" --top_k 5
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k 5
python -m src.app.cli --method bm25 --query "điều kiện kết hôn" --top_k none --threshold 1.5 --display_limit 20
python -m src.evaluation.evaluate --method all --top_k 10 --max_queries 100
python -m src.evaluation.evaluate --method all --top_k none --threshold 1.5 --max_queries 100
```

## Test

```powershell
python -m pytest tests -q
```

## Cấu trúc file chính

```text
src/data/prepare_data.py
src/indexing/inverted_index.py
src/indexing/build_retrieval_models.py
src/retrieval/boolean_retrieval.py
src/retrieval/tfidf_retrieval.py
src/retrieval/bm25_retrieval.py
src/app/cli.py
src/evaluation/metrics.py
src/evaluation/evaluate.py
```
