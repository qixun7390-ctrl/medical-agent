import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ==================== 新增：数值提取器 ====================
@dataclass
class NumericValue:
    """提取到的数值信息"""
    value: float
    raw_text: str
    position: int
    context: str
    possible_types: List[Dict[str, Any]]
    resolved_type: Optional[str] = None
    unit: Optional[str] = None
    confidence: float = 0.0
    warning: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def is_ambiguous(self) -> bool:
        return len(self.possible_types) > 1 and self.resolved_type is None


class NumericExtractor:
    """医学数值提取器 - 解决"38度"被误认为"血小板38"的问题"""

    # 医学数值类型定义
    NUMERIC_TYPES = {
        "temperature": {
            "name": "体温",
            "patterns": ["体温", "温度", "发烧", "发热", "度", "℃", "°C", "摄氏度"],
            "unit": "℃",
            "range": (35.0, 42.0),
            "warn_range": (34.0, 44.0),
            "keywords": ["体温", "温度", "发烧", "发热"]
        },
        "platelet": {
            "name": "血小板",
            "patterns": ["血小板", "plt", "PLT", "血常规"],
            "unit": "×10^9/L",
            "range": (100, 450),
            "warn_range": (50, 600),
            "keywords": ["血小板", "血常规"]
        },
        "blood_pressure_systolic": {
            "name": "收缩压",
            "patterns": ["收缩压", "高压", "上压"],
            "unit": "mmHg",
            "range": (90, 140),
            "warn_range": (70, 200),
            "keywords": ["血压"]
        },
        "blood_pressure_diastolic": {
            "name": "舒张压",
            "patterns": ["舒张压", "低压", "下压"],
            "unit": "mmHg",
            "range": (60, 90),
            "warn_range": (40, 120),
            "keywords": ["血压"]
        },
        "heart_rate": {
            "name": "心率",
            "patterns": ["心率", "脉搏", "心跳"],
            "unit": "次/分",
            "range": (60, 100),
            "warn_range": (40, 180),
            "keywords": ["心率", "心跳"]
        }
    }

    CONTEXT_WINDOW = 20

    @classmethod
    def extract(cls, text: str) -> List[NumericValue]:
        """从文本中提取所有数值并判断可能含义"""
        results = []
        pattern = r'(\d+\.?\d*)'

        for match in re.finditer(pattern, text):
            value = float(match.group())
            position = match.start()

            # 获取上下文
            start = max(0, position - cls.CONTEXT_WINDOW)
            end = min(len(text), position + cls.CONTEXT_WINDOW)
            context = text[start:end]

            # 判断可能的类型
            possible_types = cls._identify_possible_types(value, context)

            results.append(NumericValue(
                value=value,
                raw_text=match.group(),
                position=position,
                context=context,
                possible_types=possible_types
            ))

        return results

    @classmethod
    def _identify_possible_types(cls, value: float, context: str) -> List[Dict]:
        """根据数值和上下文判断可能的医学类型"""
        possible = []
        context_lower = context.lower()

        for type_key, type_info in cls.NUMERIC_TYPES.items():
            min_warn, max_warn = type_info["warn_range"]
            if not (min_warn <= value <= max_warn):
                continue

            has_keyword = any(kw in context_lower for kw in type_info["keywords"])
            min_norm, max_norm = type_info["range"]

            if min_norm <= value <= max_norm or has_keyword:
                possible.append({
                    "type": type_key,
                    "name": type_info["name"],
                    "unit": type_info["unit"],
                    "is_normal": min_norm <= value <= max_norm,
                    "confidence": 0.8 if has_keyword else 0.5,
                    "range": type_info["range"]
                })

        return possible

    @classmethod
    def resolve_ambiguity(cls, numeric: NumericValue, full_context: str = "") -> NumericValue:
        """解决数值歧义，确定最终类型"""
        if len(numeric.possible_types) == 1:
            numeric.resolved_type = numeric.possible_types[0]["type"]
            numeric.unit = numeric.possible_types[0]["unit"]
            numeric.confidence = numeric.possible_types[0]["confidence"]

        elif len(numeric.possible_types) > 1:
            context = full_context or numeric.context

            # 优先匹配明确的单位关键词
            for pt in numeric.possible_types:
                type_info = cls.NUMERIC_TYPES[pt["type"]]
                if any(p in context for p in type_info["patterns"][:3]):  # 主要模式
                    numeric.resolved_type = pt["type"]
                    numeric.unit = pt["unit"]
                    numeric.confidence = 0.9
                    break

            # 如果还没确定，选置信度最高的
            if not numeric.resolved_type:
                best = max(numeric.possible_types, key=lambda x: x["confidence"])
                numeric.resolved_type = best["type"]
                numeric.unit = best["unit"]
                numeric.confidence = best["confidence"]
                numeric.warning = f"数值{numeric.value}可能指{', '.join([p['name'] for p in numeric.possible_types])}，已按{best['name']}处理"

        return numeric


# ==================== 扩展RuleInput，添加数值歧义规则类型 ====================
class RuleInput(BaseModel):
    """
    name:规则唯一标识
    type:规则类型：numeric | keyword | disambiguation
    ------
    数值型专用字段:
    slot:要检查的字段
    op:">="
    value:阈值
    ------
    关键词型专用字段:
    keywords:['儿童','小儿','宝宝']
    ------
    数值歧义规则专用字段:
    ambiguous_ranges: 歧义数值范围定义
    ------
    共同字段:
    add_queries:补充检索问题
    guidance:规则建议
    """
    name: str = Field(..., description="Rule name")
    type: str = Field(..., description="numeric | keyword | disambiguation")

    # 数值型字段
    slot: Optional[str] = None
    op: Optional[str] = None
    value: Optional[float] = None

    # 关键词型字段
    keywords: Optional[List[str]] = None

    # 数值歧义规则字段
    ambiguous_ranges: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="歧义数值范围，如：[{'type':'temperature','min':35,'max':42,'unit':'℃'}]"
    )

    # 共同字段
    add_queries: List[str] = Field(default_factory=list)
    guidance: Optional[str] = None


# ==================== 新增：规则引擎核心类 ====================
class RuleEngine:
    """规则引擎 - 执行数值提取和规则匹配"""

    def __init__(self):
        self.extractor = NumericExtractor()
        self.rules_data = _load_rules()
        self.rules = self.rules_data.get("rules", [])

    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询，返回规则执行结果
        """
        result = {
            "extracted_numerics": [],
            "ambiguous_values": [],
            "rule_hits": [],
            "guidance": [],
            "need_clarification": False,
            "clarification_question": None
        }

        # 1. 提取所有数值
        numerics = self.extractor.extract(query)
        result["extracted_numerics"] = [n.to_dict() for n in numerics]

        # 2. 检查数值歧义
        for num in numerics:
            resolved = self.extractor.resolve_ambiguity(num, query)
            if resolved.warning:
                result["ambiguous_values"].append(resolved.to_dict())

            # 如果数值歧义且无法解决，需要澄清
            if resolved.is_ambiguous():
                result["need_clarification"] = True
                possible_names = [p["name"] for p in resolved.possible_types]
                result["clarification_question"] = (
                    f"您提到的{resolved.raw_text}是指"
                    f"{'、'.join(possible_names)}？请说明具体是哪项指标。"
                )
                return result

        # 3. 应用规则
        for num in numerics:
            if num.resolved_type:
                self._apply_numeric_rules(num, result)

        self._apply_keyword_rules(query, result)

        # 4. 去重guidance
        result["guidance"] = list(set(result["guidance"]))
        result["guidance"] = "；".join(result["guidance"]) if result["guidance"] else ""

        return result

    def _apply_numeric_rules(self, num: NumericValue, result: Dict):
        """应用数值型规则"""
        for rule in self.rules:
            if rule.get("type") != "numeric":
                continue
            if rule.get("slot") != num.resolved_type:
                continue

            # 比较数值
            op = rule.get("op")
            threshold = rule.get("value")
            hit = False

            if op == ">=" and num.value >= threshold:
                hit = True
            elif op == "<=" and num.value <= threshold:
                hit = True
            elif op == ">" and num.value > threshold:
                hit = True
            elif op == "<" and num.value < threshold:
                hit = True
            elif op == "==" and abs(num.value - threshold) < 0.1:
                hit = True

            if hit:
                result["rule_hits"].append(rule["name"])
                if rule.get("guidance"):
                    result["guidance"].append(rule["guidance"])
                if rule.get("add_queries"):
                    result.setdefault("add_queries", []).extend(rule["add_queries"])

    def _apply_keyword_rules(self, query: str, result: Dict):
        """应用关键词型规则"""
        for rule in self.rules:
            if rule.get("type") != "keyword":
                continue
            keywords = rule.get("keywords", [])
            if any(kw in query for kw in keywords):
                result["rule_hits"].append(rule["name"])
                if rule.get("guidance"):
                    result["guidance"].append(rule["guidance"])
                if rule.get("add_queries"):
                    result.setdefault("add_queries", []).extend(rule["add_queries"])


# ==================== 原有的规则管理代码保持不变 ====================
def _default_rules_path() -> str:
    base = Path(__file__).resolve().parents[2]
    candidate = base / "medical-agent" / "configs" / "clinical_rules.json"
    if candidate.exists():
        return str(candidate)
    return str(base / "configs" / "clinical_rules.json")


RULES_PATH = os.environ.get("CLINICAL_RULES_PATH", _default_rules_path())


def _load_rules() -> Dict[str, Any]:
    if not os.path.exists(RULES_PATH):
        return {"version": 1, "rules": []}
    text = Path(RULES_PATH).read_text(encoding="utf-8", errors="ignore")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return json.loads(text) if text.strip() else {"version": 1, "rules": []}


def _save_rules(data: Dict[str, Any]) -> None:
    Path(RULES_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(RULES_PATH).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_rules_data() -> Dict[str, Any]:
    return _load_rules()


def add_rule(payload: RuleInput) -> Dict[str, Any]:
    data = _load_rules()
    rules = data.get("rules", [])

    if payload.type not in {"numeric", "keyword", "disambiguation"}:
        raise ValueError("type must be numeric, keyword, or disambiguation")

    if payload.type == "numeric" and (payload.slot is None or payload.op is None or payload.value is None):
        raise ValueError("numeric rule requires slot, op, value")

    if payload.type == "keyword" and (not payload.keywords):
        raise ValueError("keyword rule requires keywords")

    rule = payload.model_dump()
    rules.append(rule)
    data["rules"] = rules
    _save_rules(data)
    return {"status": "ok", "rules_count": len(rules)}


def render_rules_ui() -> HTMLResponse:
    data = _load_rules()
    pretty = json.dumps(data, ensure_ascii=False, indent=2)
    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Clinical Rules Editor</title>
        <style>
          body {{ font-family: Arial, sans-serif; padding: 20px; }}
          textarea {{ width: 100%; height: 260px; }}
          input, textarea {{ margin: 6px 0; }}
          .row {{ margin-bottom: 10px; }}
          .hint {{ color: #666; font-size: 12px; }}
        </style>
      </head>
      <body>
        <h2>Clinical Rules Editor</h2>
        <div class="row">
          <button onclick="refresh()">Refresh</button>
        </div>
        <div class="row">
          <h3>Current Rules (JSON)</h3>
          <textarea id="rules">{pretty}</textarea>
          <div class="hint">仅用于查看。新增规则请使用下面表单。</div>
        </div>
        <div class="row">
          <h3>Add Rule</h3>
          <input id="name" placeholder="name" />
          <input id="type" placeholder="numeric | keyword | disambiguation" />
          <input id="slot" placeholder="slot (e.g., temperature)" />
          <input id="op" placeholder="op (>=, <=, >, <, ==)" />
          <input id="value" placeholder="value (e.g., 39.0)" />
          <input id="keywords" placeholder="keywords (comma separated)" />
          <input id="add_queries" placeholder="add_queries (comma separated)" />
          <input id="guidance" placeholder="guidance" />
          <button onclick="addRule()">Add</button>
          <div id="msg" class="hint"></div>
        </div>
        <script>
          async function refresh() {{
            const r = await fetch('/api/rules');
            const j = await r.json();
            document.getElementById('rules').value = JSON.stringify(j, null, 2);
          }}
          async function addRule() {{
            const payload = {{
              name: document.getElementById('name').value,
              type: document.getElementById('type').value,
              slot: document.getElementById('slot').value || null,
              op: document.getElementById('op').value || null,
              value: document.getElementById('value').value ? parseFloat(document.getElementById('value').value) : null,
              keywords: document.getElementById('keywords').value
                ? document.getElementById('keywords').value.split(',').map(s => s.trim()).filter(Boolean)
                : null,
              add_queries: document.getElementById('add_queries').value
                ? document.getElementById('add_queries').value.split(',').map(s => s.trim()).filter(Boolean)
                : [],
              guidance: document.getElementById('guidance').value || null
            }};
            const r = await fetch('/api/rules', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload)
            }});
            const j = await r.json();
            document.getElementById('msg').innerText = JSON.stringify(j);
            await refresh();
          }}
          refresh();
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


# ==================== 导出便捷接口 ====================
rule_engine = RuleEngine()


def process_with_rules(query: str) -> Dict[str, Any]:
    """对外提供的规则处理接口"""
    return rule_engine.process(query)