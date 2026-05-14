# Pure Python IR Audit

Date: 2026-05-13

## SAFE imports

- `json`, `math`, `re`, `pickle`, `argparse`, `pathlib`, `collections`
- `tqdm` for progress display only
- `datasets` for loading the legal dataset only
- `pandas`, `streamlit`, `pytest` for demo UI and testing only

## Imports removed or requiring removal

- `src/preprocessing/tokenizer.py`
  - Removed dynamic import `from underthesea import word_tokenize`
  - Reason: this delegates the core tokenization algorithm to an external NLP library
- `requirements.txt`
  - Removed `underthesea`
  - Removed `scikit-learn`
  - Reason: neither should be required for the current handwritten IR pipeline

## ACCEPTABLE modules after audit

- `src/data/prepare_data.py`
  - uses `datasets` only to load data
- `src/indexing/inverted_index.py`
  - inverted index built manually with `dict`, `set`, loops
- `src/retrieval/boolean_retrieval.py`
  - Boolean logic implemented manually with set operations
- `src/retrieval/tfidf_retrieval.py`
  - TF, IDF, TF-IDF vectors, postings, and cosine similarity implemented manually
- `src/retrieval/bm25_retrieval.py`
  - BM25 scoring formula implemented manually
- `src/evaluation/metrics.py`
  - `precision@k`, `recall@k`, `MAP`, `DCG`, `nDCG` implemented manually

## Files changed

- `src/preprocessing/tokenizer.py`
- `src/preprocessing/stopwords.py`
- `src/retrieval/tfidf_retrieval.py`
- `src/retrieval/bm25_retrieval.py`
- `src/data/prepare_data.py`
- `src/indexing/inverted_index.py`
- `src/app/cli.py`
- `src/evaluation/evaluate.py`
- `tests/test_preprocessing.py`
- `requirements.txt`
- `README.md`

## Handwritten algorithm status

- Preprocessing: pure Python `lowercase + regex clean + split whitespace`
- Inverted index: handwritten
- Boolean Retrieval: handwritten
- TF-IDF: handwritten
- Cosine similarity: handwritten
- BM25: handwritten
- Evaluation metrics: handwritten

## Remaining non-pure-Python areas

- `src/data/load_dataset.py` still uses `datasets`, but only for dataset loading, not for IR logic
- `src/app/demo_streamlit.py` still uses `pandas` and `streamlit`, but only for UI/report display

## Verification

- `python -m pytest tests -q`
- `python src/data/prepare_data.py`
- `python src/indexing/inverted_index.py`
- `python src/app/cli.py --query "điều kiện kết hôn" --top_k 3`
- `python src/evaluation/evaluate.py`
