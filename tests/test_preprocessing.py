from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.tokenizer import tokenize, preprocess


def test_clean_text_lowercase_and_remove_punctuation():
    text = "Xin Chào!!!"
    assert clean_text(text) == "xin chào"


def test_tokenize_not_empty():
    tokens = tokenize("Điều kiện kết hôn")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_preprocess_not_empty():
    tokens = preprocess("Người lao động được nghỉ hằng năm bao nhiêu ngày?")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_preprocess_remove_compound_stopword():
    tokens = preprocess("Người lao động được nghỉ hằng năm bao nhiêu ngày?")
    assert "bao_nhiêu" not in tokens