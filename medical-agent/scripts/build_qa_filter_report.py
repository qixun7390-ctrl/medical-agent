import json
import os
from pathlib import Path
import re
from collections import Counter, defaultdict
from json import JSONDecodeError

QA_JSONL = Path(os.environ.get("QA_JSONL", "D:\\L4阶段资料\\39-大模型微调实例-25.9.30-景南老师\\P03_v2\\P03\\ex4_meddata_10000.jsonl"))
OUT_REPORT = Path(os.environ.get("OUT_REPORT", "E:\\PythonProject2\\medical-agent\\data\\report\\qa_filter_report.md"))
OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def _cn_len(t: str) -> int:
    if not t:
        return 0
    return sum(1 for ch in t if "\u4e00" <= ch <= "\u9fff" or ch.isalnum())

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


def reason_for_low_quality(q: str, a: str) -> str | None:
    q = _norm_text(q)
    a = _norm_text(a)
    if not q or not a:
        return "missing_q_or_a"
    if _cn_len(a) < 15:
        return "answer_too_short"
    if LOW_QUALITY_ANS_RE.search(a):
        return "generic_or_template_answer"
    if "挂号" in a or "预约" in a or ("门诊" in a and _cn_len(a) < 25):
        return "scheduling_only"
    return None


def _open_qa_file(path: Path):
    try:
        return path.open("r", encoding="gb18030", errors="ignore")
    except Exception:
        return path.open("r", encoding="utf-8", errors="ignore")


total = 0
kept = 0
bad_json = 0
reasons = Counter()
examples = defaultdict(list)

with _open_qa_file(QA_JSONL) as f:
    for line in f:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except JSONDecodeError:
            bad_json += 1
            continue
        q = obj.get("question") or obj.get("query") or obj.get("q") or ""
        a = obj.get("answer") or obj.get("response") or obj.get("a") or ""

        if not q and "messages" in obj:
            messages = obj.get("messages", [])
            last_user = None
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    last_user = content
                elif role == "assistant" and last_user:
                    total += 1
                    reason = reason_for_low_quality(last_user, content)
                    if reason:
                        reasons[reason] += 1
                        if len(examples[reason]) < 3:
                            examples[reason].append((last_user, content))
                    else:
                        kept += 1
                    last_user = None
            continue

        if not q or not a:
            total += 1
            reasons["missing_q_or_a"] += 1
            if len(examples["missing_q_or_a"]) < 3:
                examples["missing_q_or_a"].append((q, a))
            continue

        total += 1
        reason = reason_for_low_quality(q, a)
        if reason:
            reasons[reason] += 1
            if len(examples[reason]) < 3:
                examples[reason].append((q, a))
        else:
            kept += 1

filtered = total - kept
kept_rate = (kept / total) if total else 0.0

lines = []
lines.append("# QA 过滤报告\n")
lines.append(f"- 源数据：`{QA_JSONL}`\n")
lines.append(f"- JSON 解析失败行数：{bad_json}\n")
lines.append(f"- 总条目：{total}\n")
lines.append(f"- 保留：{kept} ({kept_rate:.1%})\n")
lines.append(f"- 过滤：{filtered} ({1-kept_rate:.1%})\n")
lines.append("\n## 过滤原因统计\n")
for reason, cnt in reasons.most_common():
    lines.append(f"- {reason}: {cnt}\n")

lines.append("\n## 典型样例（每类最多3条）\n")
for reason, exs in examples.items():
    lines.append(f"\n### {reason}\n")
    for q, a in exs:
        q = _norm_text(q)
        a = _norm_text(a)
        lines.append(f"- 问：{q}\n  答：{a}\n")

OUT_REPORT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote report to {OUT_REPORT}")
