VIETNAMESE_STOPWORDS = {
    # Từ chức năng phổ biến
    "là", "và", "của", "có", "cho", "trong", "một", "các", "những",
    "được", "theo", "với", "khi", "thì", "này", "đó", "về", "từ",
    "tại", "ra", "vào", "hay", "hoặc",

    # Từ hỏi thường gây nhiễu trong query
    "bao", "nhiêu", "bao_nhiêu", "gì", "nào", "như_thế_nào",
    "ai", "đâu", "khi_nào",

    # Một số từ rất chung
    "quy_định", "pháp_luật",
}


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token and token not in VIETNAMESE_STOPWORDS
    ]