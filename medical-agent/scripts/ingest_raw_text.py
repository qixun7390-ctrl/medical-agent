import json
import os
import re
from pathlib import Path

RAW_TXT = Path(os.environ.get("RAW_TXT", "D:\\L4阶段资料\\39-大模型微调实例-25.9.30-景南老师\\P03_v2\\P03\\ex2_medtext.txt"))
OUT = Path(os.environ.get("OUT", "E:\\PythonProject2\\medical-agent\\data\\processed\\raw_corpus.jsonl"))
OUT.parent.mkdir(parents=True, exist_ok=True)

# Simple noise filters
MIN_LEN = 80
BOILERPLATE_PATTERNS = [
    r"免责声明",
    r"仅供参考",
    r"版权",
    r"转载",
]

boiler_re = re.compile("|".join(BOILERPLATE_PATTERNS))

seen = set()

with RAW_TXT.open("r", encoding="utf-8", errors="ignore") as f_in, OUT.open("w", encoding="utf-8") as f_out:
    text = f_in.read()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    kept = 0
    for i, c in enumerate(chunks):
        c = re.sub(r"\s+", " ", c)
        if len(c) < MIN_LEN:
            continue
        if boiler_re.search(c):
            continue
        h = hash(c)
        if h in seen:
            continue
        seen.add(h)
        kept += 1
        obj = {"doc_id": f"raw_{i}", "text": c, "source": str(RAW_TXT)}
        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {kept} cleaned raw records to {OUT}")
