from functools import lru_cache

from src.preprocessing.stopwords import remove_stopwords
from src.preprocessing.text_cleaner import clean_text
from src.utils.console import configure_utf8_stdout


def tokenize(text: str, use_underthesea: bool = True) -> list[str]:
    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    if use_underthesea:
        try:
            from underthesea import word_tokenize

            tokenized_text = word_tokenize(cleaned_text, format="text")
            tokens = tokenized_text.split()
            if tokens:
                return tokens
        except Exception:
            pass

    return cleaned_text.split()


# Full-data builds reuse the same corpus across index, TF-IDF, and BM25.
# A larger cache avoids repeating tokenization work across those passes.
@lru_cache(maxsize=200000)
def preprocess_cached(
    text: str,
    use_underthesea: bool = True,
    remove_stopword: bool = True,
) -> tuple[str, ...]:
    tokens = tokenize(text, use_underthesea=use_underthesea)

    if remove_stopword:
        tokens = remove_stopwords(tokens)

    return tuple(tokens)


def preprocess(
    text: str,
    use_underthesea: bool = True,
    remove_stopword: bool = True,
) -> list[str]:
    return list(preprocess_cached(text, use_underthesea, remove_stopword))


if __name__ == "__main__":
    configure_utf8_stdout()
    sample_text = "Người lao động được nghỉ hằng năm bao nhiêu ngày?"
    print("Original:", sample_text)
    print("Tokens:", tokenize(sample_text))
    print("Preprocessed:", preprocess(sample_text))
