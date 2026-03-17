import re
from typing import List, Dict
from app.agent.state import AgentState
from app.core.config import settings


def estimate_tokens(text: str) -> int:
    """
    估算token，精准管理上下文的长度
    :param text
    :return: "the length of text"
    """
    # Rough token estimation: Chinese ~ 1 char per token, English ~ 4 chars per token
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    non = len(text) - chinese
    return chinese + max(0, non // 4)


def _join_with_budget(lines: List[str], max_tokens: int) -> str:
    """
    按照max token将多行进行文本进行拼接
    param:
        lines
        max_tokens
    :return:"拼接后的文本"
    """
    out = []
    used = 0
    for line in lines:
        t = estimate_tokens(line)
        if used + t > max_tokens:
            break
        out.append(line)
        used += t
    return "\n".join(out)


def _compress_history(history: List[Dict[str, str]], max_tokens: int) -> str:
    """
    把长对话压缩成简短摘要
    param
        history
        max_tokens
    return:
    """
    # Heuristic compression: keep salient user facts and assistant advice
    if not history:
        return ""
    #按照QA对处理历史病例
    qa_pairs = []
    for i in range(0,len(history),2):
        if i+1 < len(history):
            qa_pairs.append({
                "user":history[i].get("content","").strip(),
                "assistant":history[i+1].get("content","").strip()
            })

    keywords = ["发热", "咳嗽", "抽动", "疼痛", "用药", "检查", "诊断", "症状", "既往", "过敏", "血", "胸", "呼吸", "腹痛", "呕吐", "腹泻","严重","发热","天","不止","不停"]
    valid_pairs = [pair for pair in qa_pairs if any(k in pair["user"] for k in keywords)]
    if not valid_pairs:
        valid_pairs = qa_pairs[-3:]
    #构建摘要
    summary_part = []
    for i,pair in enumerate(valid_pairs[:3]):
        case = f"病例{i+1}: 病症{pair['user'][:30]}... ->建议:{pair['assistant'][:50]}..."
        summary_part.append(case)
    summary = "|".join(summary_part)
    if estimate_tokens(summary) > max_tokens:
        summary = summary[:max(20,max_tokens * 2)]
    return summary

class ContextEngineer:
    def __init__(self):
        """
        初始化
        将系统指令，证据，历史，总输入的token预算初始化
        """
        self.system_budget = settings.context_system_budget
        self.evidence_budget = settings.context_evidence_budget
        self.history_budget = settings.context_history_budget
        self.input_budget = settings.context_input_budget
    #AgentState - langgraph的全局状态机
    def build_prompt(self, state: AgentState, evidences: List[str], force_citations: bool = False) -> str:
        system = (
            "你是医疗健康助手。回答必须基于证据，不得编造。"
            "如果证据不足，明确说明并建议就医。"
        )

        history_lines = [f"{m.get('role','')}: {m.get('content','')}" for m in state.history]
        # Split history into recent window + summary for older parts
        recent = []
        used = 0
        #从最新消息开始
        for line in reversed(history_lines):
            t = estimate_tokens(line)
            #不能超过预算
            if used + t > self.history_budget:
                break
            recent.append(line)
            used += t
        #恢复正序
        recent = list(reversed(recent))
        older = history_lines[: max(0, len(history_lines) - len(recent))]

        #对于过往会话的记忆进行压缩
        summary = _compress_history(older, max_tokens=min(200, self.history_budget // 2))
        #合并之前会话的摘要（episodic记忆）-结构化记忆(跨会话关键信息摘要)
        if state.episodic_summary:
            summary = (state.episodic_summary + " | " + summary).strip(" | ")

        history_block = ""
        if summary:
            history_block += f"[历史摘要]\n{summary}\n"
        if recent:
            history_block += "[近期对话]\n" + "\n".join(recent) + "\n"

        # Rule summary (structured signals must be respected)
        rule_block = ""
        if state.extracted_slots:
            rule_block += f"[结构化字段]\n{state.extracted_slots}\n"
        if state.rule_hits:
            rule_block += f"[规则命中]\n{', '.join(state.rule_hits)}\n"
        if state.guidance:
            rule_block += f"[规则建议]\n{state.guidance}\n"

        # Evidence block with budget
        evidence_text = _join_with_budget(evidences, self.evidence_budget)

        cite_rule = "每句话末尾必须带引用，如 [doc_id]。" if force_citations else "关键结论必须带引用。"
        refusal = "如果证据不足，明确说明无法安全回答。"

        prompt = (
            f"[系统指令]\n{system}\n\n"

            f"[历史对话]\n{history_block}\n\n"

            f"[规则引擎输出]\n{rule_block}\n\n"

            f"[检索证据]\n{evidence_text}\n\n"

            f"[当前问题]\n{state.query}\n\n"

            "[推理过程]\n"
            "请严格按以下思维链（Chain of Thought）步骤进行推理，每一步都要写出你的思考过程：\n\n"

            "=== 步骤1：关键信息提取 ===\n"
            "请从历史对话和规则引擎输出中，提取所有关键信息，并按类别整理：\n"
            "• 患者基本信息（年龄、性别等）：\n"
            "• 主要症状和体征：\n"
            "• 症状持续时间：\n"
            "• 严重程度指标（体温、疼痛评分等）：\n"
            "• 既往病史/过敏史：\n"
            "• 已采取的医疗措施：\n"
            "• 其他重要信息：\n\n"

            "=== 步骤2：证据分析 ===\n"
            "分析每条检索到的证据，思考：\n"
            "• [证据1]支持什么？权威性如何？\n"
            "• [证据2]支持什么？权威性如何？\n"
            "• [证据3]支持什么？权威性如何？\n"
            "• 这些证据之间是否一致？有无矛盾？\n"
            "• 基于证据，可以得出哪些初步判断？\n\n"

            "=== 步骤3：规则应用 ===\n"
            "分析规则引擎的输出，思考：\n"
            "• 触发了哪些规则？为什么触发？\n"
            "• 这些规则提示了什么风险？\n"
            "• 规则建议如何影响诊断和治疗决策？\n"
            "• 有哪些必须遵守的安全准则？\n\n"

            "=== 步骤4：综合推理 ===\n"
            "结合以上所有信息，进行综合分析：\n"
            "• 最可能的诊断/问题是什么？（列出支持证据和规则）\n"
            "• 还有哪些鉴别诊断需要考虑？（列出可能性及依据）\n"
            "• 存在哪些风险需要警惕？（基于规则和证据）\n"
            "• 还需要哪些信息来确认诊断？\n"
            "• 目前情况是否紧急？是否需要立即就医？\n\n"

            "=== 步骤5：治疗方案构建 ===\n"
            "基于推理结果，构建分层治疗方案：\n"
            "• 家庭护理措施有哪些？\n"
            "• 药物治疗建议（如有）？\n"
            "• 就医指征是什么？\n"
            "• 需要随访观察什么？\n\n"

            "请将以上思考过程完整写出来，然后再给出最终回答。\n\n"

            "[最终回答]\n"
            "请按以下结构化格式输出你的最终答案，确保每个结论都有证据或规则支持：\n\n"

            "[简要结论]\n"
            "（用1-2句话概括核心结论，必须引用证据，格式如[证据1]，同时说明紧急程度）\n\n"

            "[可能性分析]\n"
            "（按可能性从高到低列出最多3条，每条必须包含：可能性描述、置信度、依据）\n"
            "1. [可能性] - 置信度[高/中/低] - 依据：[引用证据和规则]  \n"
            "2. [可能性] - 置信度[高/中/低] - 依据：[引用证据和规则]  \n"
            "3. [可能性] - 置信度[高/中/低] - 依据：[引用证据和规则]  \n\n"

            "[治疗建议]\n"
            "（分层次给出具体可操作的建议）\n"
            "• 家庭护理：\n"
            "• 药物治疗：\n"
            "• 就医指征：\n"
            "• 随访观察：\n\n"

            "[注意事项]\n"
            "（列出所有需要警惕的风险和禁忌）\n"
            "• 风险警示：\n"
            "• 禁忌事项：\n"
            "• 需要密切观察的症状：\n\n"

            "[补充说明]\n"
            "（如果有信息不足或不确定的地方，在这里说明）\n\n"

            f"{cite_rule}\n\n"
            f"{refusal}"
        )
        return prompt
