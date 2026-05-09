import re
import unicodedata


_SPECIAL_CHARS_PATTERN = re.compile(
    r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
    r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡ"
    r"ùúụủũưừứựửữỳýỵỷỹđ]",
    flags=re.UNICODE,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    - ép về string
    - chuẩn hóa Unicode NFC
    - lowercase
    - bỏ ký tự đặc biệt
    - chuẩn hóa khoảng trắng
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    text = _SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    return text