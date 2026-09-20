from __future__ import annotations

import re

from localpilot.schemas import Intent


def _contains_any(text: str, terms: tuple) -> bool:
    return any(term in text for term in terms)


def parse_intent(text: str) -> Intent:
    lowered = text.lower()

    if _contains_any(lowered, ("代码", "编程", "coding", "code review", "code-review")):
        task = "coding"
    elif _contains_any(lowered, ("embedding", "向量", "嵌入")):
        task = "embedding"
    else:
        task = "chat"

    if _contains_any(lowered, ("速度优先", "响应速度", "低延迟", "latency", "fast")):
        priority = "latency"
    elif _contains_any(lowered, ("质量优先", "最高质量", "quality first")):
        priority = "quality"
    elif _contains_any(lowered, ("低内存", "省内存", "low memory")):
        priority = "low_memory"
    else:
        priority = "balanced"

    local_terms = (
        "本地",
        "不能上传",
        "不能离开",
        "隐私",
        "local",
        "offline",
        "private",
    )
    privacy = "local_only" if _contains_any(lowered, local_terms) else "local_preferred"

    context_length = 8192
    context_match = re.search(r"(\d+)\s*[kK]\s*(?:context|上下文)?", text)
    if context_match:
        context_length = int(context_match.group(1)) * 1024

    languages = ["zh", "en"] if re.search(r"[\u4e00-\u9fff]", text) else ["en"]
    capabilities = []
    if task == "coding":
        capabilities.append("code_review")
    if _contains_any(lowered, ("tool", "工具调用", "function calling")):
        capabilities.append("tool_calling")

    return Intent(
        task=task,
        privacy=privacy,
        priority=priority,
        quality="high",
        context_length=context_length,
        concurrency=1,
        preferred_language=languages,
        capabilities=capabilities,
        raw_text=text,
    )

