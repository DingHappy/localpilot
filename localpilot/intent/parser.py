from __future__ import annotations

import re

from localpilot.schemas import Intent


TASK_TERMS = (
    ("coding", ("代码", "编程", "coding", "code review", "code-review", "程序")),
    ("agentic", ("agent", "智能体", "工作流", "workflow", "tool use", "工具调用")),
    ("vision", ("图片", "图像", "视觉", "截图", "vision", "image", "ocr", "文档识别")),
    ("audio", ("语音", "音频", "说话", "voice", "audio", "speech", "asr", "tts")),
    ("embedding", ("embedding", "向量", "嵌入", "检索", "retrieval", "rag")),
)

PRIORITY_TERMS = (
    ("latency", ("速度优先", "响应速度", "低延迟", "延迟", "latency", "fast", "快")),
    (
        "throughput",
        ("吞吐", "并发", "批量", "throughput", "concurrent", "batch", "多用户"),
    ),
    ("quality", ("质量优先", "最高质量", "效果最好", "quality first", "最强")),
    ("low_memory", ("低内存", "省内存", "内存有限", "low memory", "small footprint")),
    ("long_context", ("长上下文", "长文本", "大上下文", "long context", "整个仓库")),
)

LOCAL_TERMS = (
    "本地",
    "不能上传",
    "不能离开",
    # Matches 不出网 and 不能出网 alike; the negation is already implied by
    # the phrase, so keying on the shorter stem avoids listing variants.
    "出网",
    "隐私",
    "内网",
    "离线",
    "local",
    "offline",
    "private",
    "on-premise",
    "air-gap",
)

MODALITY_TERMS = (
    ("image", ("图片", "图像", "视觉", "截图", "image", "vision", "ocr")),
    ("audio", ("语音", "音频", "voice", "audio", "speech")),
    ("document", ("pdf", "文档", "document", "扫描件")),
)


def _contains_any(text: str, terms: tuple) -> bool:
    return any(term in text for term in terms)


def _first_match(text: str, table: tuple, default: str) -> str:
    for value, terms in table:
        if _contains_any(text, terms):
            return value
    return default


def _context_length(text: str) -> int:
    match = re.search(r"(\d+)\s*([kKmM])\s*(?:context|上下文|tokens?)?", text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        return amount * (1024 if unit == "k" else 1024 * 1024)
    if _contains_any(text.lower(), ("长上下文", "整个仓库", "long context")):
        return 131072
    return 8192


def _concurrency(text: str) -> int:
    match = re.search(
        r"(\d+)\s*(?:个?(?:人|用户|并发|路)|users?|concurrent|requests?)", text
    )
    if match:
        return max(1, min(256, int(match.group(1))))
    return 1


def parse_intent(text: str) -> Intent:
    lowered = text.lower()

    task = _first_match(lowered, TASK_TERMS, "chat")
    priority = _first_match(lowered, PRIORITY_TERMS, "balanced")
    privacy = "local_only" if _contains_any(lowered, LOCAL_TERMS) else "local_preferred"
    context_length = _context_length(text)
    concurrency = _concurrency(lowered)

    # An explicit concurrency figure is a stronger statement of intent than
    # an adjective, so it decides the priority when the two disagree.
    if concurrency > 4 and priority in {"balanced", "latency"}:
        priority = "throughput"

    modalities = ["text"]
    for modality, terms in MODALITY_TERMS:
        if _contains_any(lowered, terms):
            modalities.append(modality)

    capabilities = []
    if task == "coding":
        capabilities.append("code_review")
    if task == "agentic":
        capabilities.append("tool_calling")
    if _contains_any(lowered, ("tool", "工具调用", "function calling")):
        capabilities.append("tool_calling")
    if _contains_any(lowered, ("推理", "reasoning", "思考")):
        capabilities.append("reasoning")
    if context_length >= 131072:
        capabilities.append("long_context")

    languages = ["zh", "en"] if re.search(r"[一-鿿]", text) else ["en"]

    return Intent(
        task=task,
        privacy=privacy,
        priority=priority,
        quality="high",
        context_length=context_length,
        concurrency=concurrency,
        preferred_language=languages,
        capabilities=sorted(set(capabilities)),
        modalities=sorted(set(modalities)),
        raw_text=text,
    )
