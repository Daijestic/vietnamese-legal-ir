from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.stopwords import remove_stopwords


def tokenize(text: str, use_underthesea: bool = True) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    if use_underthesea:
        try:
            from underthesea import word_tokenize
            tokenized_text = word_tokenize(text, format="text")
            return tokenized_text.split()
        except Exception:
            pass

    return text.split()


def preprocess(text: str, use_underthesea: bool = True, remove_stopword: bool = True) -> list[str]:
    tokens = tokenize(text, use_underthesea=use_underthesea)

    if remove_stopword:
        tokens = remove_stopwords(tokens)

    return tokens


if __name__ == "__main__":
    sample = "Người lao động được nghỉ hằng năm bao nhiêu ngày?"
    print(preprocess(sample))