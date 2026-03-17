import re
from typing import Dict, Any


def parse_numeric_slots(text: str) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}

    # Temperature in Celsius, e.g., 38度/38℃/38.5C
    m = re.search(r"(\d{2}(?:\.\d)?)\s*(?:度|℃|c|C)", text)
    if m:
        try:
            slots["temp_c"] = float(m.group(1))
        except Exception:
            pass

    # SpO2, e.g., 血氧92%
    m = re.search(r"(?:血氧|spo2)\D{0,3}(\d{2,3})\s*%?", text, re.IGNORECASE)
    if m:
        try:
            slots["spo2"] = int(m.group(1))
        except Exception:
            pass

    # Blood pressure, e.g., 血压 180/110
    m = re.search(r"(?:血压)\D{0,3}(\d{2,3})\s*/\s*(\d{2,3})", text)
    if m:
        try:
            slots["sbp"] = int(m.group(1))
            slots["dbp"] = int(m.group(2))
        except Exception:
            pass

    # Heart rate, e.g., 心率120
    m = re.search(r"(?:心率|心跳)\D{0,3}(\d{2,3})", text)
    if m:
        try:
            slots["hr"] = int(m.group(1))
        except Exception:
            pass

    # Respiratory rate, e.g., 呼吸30
    m = re.search(r"(?:呼吸|呼吸频率)\D{0,3}(\d{2,3})", text)
    if m:
        try:
            slots["rr"] = int(m.group(1))
        except Exception:
            pass

    return slots
