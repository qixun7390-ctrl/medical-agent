import json
import os
from pathlib import Path
import random
import re
from json import JSONDecodeError

QA_JSONL = Path(os.environ.get("QA_JSONL", "D:\\L4阶段资料\\39-大模型微调实例-25.9.30-景南老师\\P03_v2\\P03\\ex4_meddata_10000.jsonl"))
OUT_CORPUS = Path(os.environ.get("OUT_CORPUS", "E:\\PythonProject2\\medical-agent\\data\\processed\\qa_corpus.jsonl"))
OUT_EVAL = Path(os.environ.get("OUT_EVAL", "E:\\PythonProject2\\medical-agent\\data\\eval\\qa_eval.jsonl"))
OUT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)

records = []
qas = []


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _cn_len(t: str) -> int:
    if not t:
        return 0
    return sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())


def _score_text(s: str) -> int:
    # heuristic: prefer common QA markers + Chinese density
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    common = sum(s.count(x) for x in ["问", "答", "症状", "检查", "发热", "疼痛", "医生", "医院"])
    bad = s.count("�") + s.count("?")
    return cjk * 2 + common * 10 - bad * 5


def _try_fix(s: str) -> list:
    cands = [s]
    for enc in ("gbk", "gb18030", "latin1", "cp1252"):
        try:
            fixed = s.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            cands.append(fixed)
        except Exception:
            continue
    return cands


def fix_mojibake(s: str) -> str:
    # pick best candidate by heuristic score
    best = s
    best_score = _score_text(s)
    for cand in _try_fix(s):
        score = _score_text(cand)
        if score > best_score:
            best = cand
            best_score = score
    return best


LOW_QUALITY_ANS_PATTERNS = [
    r"请(介绍|说明).*(病情|情况|症状|时间|检查)",
    r"来我院|来我(们)?医院|到我院|来诊",
    r"您希望得到怎样的帮助",
    r"没有经验|抱歉",
    r"无法给您明确.*答复|很难给您明确",
    r"请到.*就诊|请就诊|请门诊",
    r"请提供.*检查|请提供.*资料",
    r"请问.*情况",
]
LOW_QUALITY_ANS_RE = re.compile("|".join(LOW_QUALITY_ANS_PATTERNS))


def is_low_quality_qa(q: str, a: str) -> bool:
    q = _norm_text(q)
    a = _norm_text(a)
    if not q or not a:
        return True
    if _cn_len(a) < 15:
        return True
    if LOW_QUALITY_ANS_RE.search(a):
        return True
    if "挂号" in a or "预约" in a or ("门诊" in a and _cn_len(a) < 25):
        return True
    return False


bad_json = 0

with QA_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except JSONDecodeError:
            bad_json += 1
            continue
        # Support standard QA fields
        q = obj.get("question") or obj.get("query") or obj.get("q") or ""
        a = obj.get("answer") or obj.get("response") or obj.get("a") or ""

        # Fix mojibake if needed
        q = fix_mojibake(q)
        a = fix_mojibake(a)

        # Support chat-style messages
        if not q and "messages" in obj:
            messages = obj.get("messages", [])
            pairs = []
            last_user = None
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    last_user = fix_mojibake(content)
                elif role == "assistant" and last_user:
                    pairs.append((last_user, fix_mojibake(content)))
                    last_user = None
            if pairs:
                # Use last pair for eval QA, and all pairs for corpus
                q, a = pairs[-1]
                last_doc_id = None
                for i, (qq, aa) in enumerate(pairs):
                    doc_id = obj.get("id") or obj.get("doc_id") or f"qa_{len(records)}_{i}"
                    if i == len(pairs) - 1:
                        last_doc_id = str(doc_id)
                    if not is_low_quality_qa(qq, aa):
                        text = f"问：{qq}\n答：{aa}"
                        records.append({"doc_id": str(doc_id), "text": text, "source": str(QA_JSONL)})
                if q and a and not is_low_quality_qa(q, a):
                    fallback_id = obj.get("id") or obj.get("doc_id") or f"qa_{len(qas)}"
                    qas.append({"question": q, "answer": a, "doc_id": last_doc_id or str(fallback_id)})
                continue

        if not q or not a:
            continue
        if is_low_quality_qa(q, a):
            continue
        doc_id = obj.get("id") or obj.get("doc_id") or f"qa_{len(records)}"
        text = f"问：{q}\n答：{a}"
        records.append({"doc_id": str(doc_id), "text": text, "source": str(QA_JSONL)})
        qas.append({"question": q, "answer": a, "doc_id": str(doc_id)})

# write corpus
with OUT_CORPUS.open("w", encoding="utf-8") as f_out:
    for r in records:
        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

# create eval set (10% sample)
random.seed(42)
random.shuffle(qas)
cut = max(1, int(0.1 * len(qas)))

with OUT_EVAL.open("w", encoding="utf-8") as f_eval:
    for r in qas[:cut]:
        f_eval.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(records)} QA corpus docs to {OUT_CORPUS}")
print(f"Wrote {cut} eval items to {OUT_EVAL}")
if bad_json:
    print(f"Skipped {bad_json} invalid JSON lines")
