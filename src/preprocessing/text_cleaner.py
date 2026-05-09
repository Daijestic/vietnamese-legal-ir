import re


def clean_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt cơ bản:
    - chuyển về lowercase
    - bỏ ký tự đặc biệt không cần thiết
    - chuẩn hóa khoảng trắng
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text