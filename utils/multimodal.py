from __future__ import annotations

from typing import Any


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _format_probabilities(probabilities: dict[str, float], disease_names: dict[str, str], top_k: int = 4) -> list[str]:
    rows = []
    for label, prob in sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)[:top_k]:
        name = disease_names.get(label, label)
        rows.append(f"{label}-{name}: {float(prob):.1%}")
    return rows


def build_ai_consult_multimodal_context(
    *,
    active_predictions: dict[str, int],
    active_probabilities: dict[str, float],
    patient_meta: dict[str, Any],
    disease_names: dict[str, str],
    graph_neighbors: dict[str, list[str]],
    form_age: Any = None,
    form_sex: Any = None,
    patient_history: str = "",
    symptoms: str = "",
    additional_info: str = "",
) -> str:
    """Build a concise multimodal context block for AI consultation only."""
    positive_labels = [label for label, value in active_predictions.items() if int(value) == 1]
    if not positive_labels and active_probabilities:
        positive_labels = [max(active_probabilities, key=active_probabilities.get)]

    age = _clean_value(patient_meta.get("age")) or _clean_value(form_age)
    sex = _clean_value(patient_meta.get("sex")) or _clean_value(form_sex)
    left_keywords = _clean_value(patient_meta.get("left_keywords"))
    right_keywords = _clean_value(patient_meta.get("right_keywords"))

    lines = ["## 多模态病例上下文"]
    if active_probabilities:
        lines.append("图像模型预测：")
        for item in _format_probabilities(active_probabilities, disease_names):
            lines.append(f"- {item}")
    else:
        lines.append("图像模型预测：当前未联动诊断页结果。")

    if positive_labels:
        label_text = "、".join(f"{label}-{disease_names.get(label, label)}" for label in positive_labels)
        lines.append(f"阳性/重点标签：{label_text}")

    patient_parts = []
    if age:
        patient_parts.append(f"年龄 {age}")
    if sex:
        patient_parts.append(f"性别 {sex}")
    lines.append("患者基础信息：" + ("；".join(patient_parts) if patient_parts else "未提供"))

    keyword_parts = []
    if left_keywords:
        keyword_parts.append(f"左眼关键词：{left_keywords}")
    if right_keywords:
        keyword_parts.append(f"右眼关键词：{right_keywords}")
    if keyword_parts:
        lines.append("CSV 诊断关键词：" + "；".join(keyword_parts))

    user_parts = []
    if _clean_value(patient_history):
        user_parts.append(f"既往病史：{patient_history}")
    if _clean_value(symptoms):
        user_parts.append(f"主诉症状：{symptoms}")
    if _clean_value(additional_info):
        user_parts.append(f"检查发现：{additional_info}")
    if user_parts:
        lines.append("用户补充信息：" + "；".join(user_parts))

    evidence_lines = []
    for label in positive_labels:
        neighbors = graph_neighbors.get(label, [])
        if neighbors:
            evidence_lines.append(f"{label}-{disease_names.get(label, label)} 关联：" + "、".join(neighbors[:6]))
    if evidence_lines:
        lines.append("知识图谱关联：")
        lines.extend(f"- {line}" for line in evidence_lines)

    lines.append("使用边界：以上信息仅用于 AI 问诊辅助解释，不覆盖图像模型原始概率，也不能替代医生诊断。")
    return "\n".join(lines)
