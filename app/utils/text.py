import re
from typing import List

# Simple medical term extractor for Chinese text
SUFFIXES = ["炎", "症", "病", "癌", "综合征", "感染", "肿瘤", "结石", "高血压", "糖尿病"]

TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")

def extract_terms(text: str) -> List[str]:
    if not text:
        return []
    candidates = TERM_RE.findall(text)
    terms = set()
    for c in candidates:
        for s in SUFFIXES:
            if c.endswith(s):
                terms.add(c)
    return list(terms)
