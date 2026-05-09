VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "có", "cho", "trong", "một", "các",
    "những", "được", "theo", "với", "khi", "thì", "này",
    "đó", "về", "từ", "tại", "ra", "vào", "hay", "hoặc",
    "bao", "nhiêu", "gì"
}


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in VIETNAMESE_STOPWORDS]