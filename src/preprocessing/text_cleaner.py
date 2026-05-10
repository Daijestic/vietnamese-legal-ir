import re
import unicodedata


_NON_WORD_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Chuan hoa van ban tieng Viet:
    - ep ve string
    - chuan hoa Unicode NFC
    - lowercase
    - bo ky tu dac biet
    - chuan hoa khoang trang
    """
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text)).lower()
    text = _NON_WORD_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
