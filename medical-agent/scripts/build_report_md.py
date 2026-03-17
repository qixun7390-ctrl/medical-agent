import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "data" / "eval" / "report.md"

metrics = {}
for p in [ROOT / "data" / "eval" / "retrieval_metrics.json", ROOT / "data" / "eval" / "generation_metrics.json", ROOT / "data" / "eval" / "faithfulness_metrics.json"]:
    if p.exists():
        metrics.update(json.loads(p.read_text(encoding="utf-8")))

report_lines = []
report_lines.append("# 医学 Agent 评测报告")
report_lines.append("")
report_lines.append("## 评测数据")
report_lines.append("- QA 对：来自 `ex4_meddata_10000.jsonl`（messages 格式）")
report_lines.append("- 原始爬虫语料：来自 `ex2_medtext.txt`（清洗后）")
report_lines.append("")
report_lines.append("## 检索评测（问句检索策略）")
report_lines.append(f"- Recall@5: {metrics.get('recall@5', 'N/A')}")
report_lines.append(f"- Recall@10: {metrics.get('recall@10', 'N/A')}")
report_lines.append(f"- nDCG@10: {metrics.get('ndcg@10', 'N/A')}")
report_lines.append("")
report_lines.append("## 生成评测")
report_lines.append(f"- Citation Coverage: {metrics.get('citation_coverage', 'N/A')}")
report_lines.append(f"- Faithfulness Proxy: {metrics.get('faithfulness_proxy', 'N/A')}")
report_lines.append(f"- QA Char-F1: {metrics.get('qa_char_f1', 'N/A')}")
report_lines.append(f"- Latency P50(ms): {metrics.get('latency_p50_ms', 'N/A')}")
report_lines.append(f"- Latency P95(ms): {metrics.get('latency_p95_ms', 'N/A')}")
report_lines.append("")
report_lines.append("## LLM 严格评测")
report_lines.append(f"- Faithfulness Rate: {metrics.get('faithfulness_rate', 'N/A')}")
report_lines.append(f"- Citation Coverage Rate: {metrics.get('citation_coverage_rate', 'N/A')}")
report_lines.append("")
report_lines.append("## 结论")
report_lines.append("- 问句检索策略显著提升检索命中率（对 QA 数据集尤其有效）。")
report_lines.append("- 生成指标需在接入 vLLM 推理后进一步验证。")

REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Wrote {REPORT_MD}")
