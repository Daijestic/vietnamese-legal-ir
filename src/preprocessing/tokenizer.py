from functools import lru_cache

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


@lru_cache(maxsize=50000)
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