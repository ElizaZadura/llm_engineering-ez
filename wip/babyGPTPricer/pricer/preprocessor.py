from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_MODEL_NAME = "ollama/llama3.1:8b"
DEFAULT_REASONING_EFFORT = "low"

# Keep product descriptions concise for downstream model training.
DEFAULT_MAX_DESCRIPTION_CHARS = 500
DEFAULT_TARGET_DESCRIPTION_CHARS = 450

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""

_SKU_PATTERN = re.compile(r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b")
_WHITESPACE_PATTERN = re.compile(r"\s+")

ATTRIBUTE_PREFIXES = (
    "material",
    "color",
    "size",
    "fit",
    "style",
    "pattern",
    "sleeve",
    "neck",
    "closure",
    "occasion",
    "fabric",
    "department",
    "gender",
    "age range",
)

QUALITY_CUE_WORDS = (
    "premium",
    "durable",
    "lightweight",
    "breathable",
    "soft",
    "comfortable",
    "waterproof",
    "stretch",
    "quality",
    "sturdy",
    "long-lasting",
)


class Preprocessor:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, reasoning_effort=DEFAULT_REASONING_EFFORT):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort

    def messages_for(self, text: str) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        from litellm import completion

        messages = self.messages_for(text)
        response = completion(
            messages=messages, model=self.model_name, reasoning_effort=self.reasoning_effort
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = _SKU_PATTERN.sub("", str(text))
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def flatten_text(value: str | Iterable[str] | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, Iterable):
        parts: list[str] = []
        for item in value:
            cleaned = normalize_text(str(item))
            if cleaned:
                parts.append(cleaned)
        return normalize_text(" ".join(parts))
    return normalize_text(str(value))


def _sentence_split(text: str) -> list[str]:
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip(" ;,.-") for p in pieces if p.strip(" ;,.-")]


def _score_sentence(sentence: str) -> int:
    s = sentence.lower()
    score = 0
    if any(w in s for w in QUALITY_CUE_WORDS):
        score += 3
    if any(s.startswith(prefix + ":") or (" " + prefix + " ") in (" " + s + " ") for prefix in ATTRIBUTE_PREFIXES):
        score += 2
    # Avoid very short generic fragments.
    if len(sentence) >= 45:
        score += 1
    return score


def prune_description(
    description: str,
    max_chars: int = DEFAULT_MAX_DESCRIPTION_CHARS,
    target_chars: int = DEFAULT_TARGET_DESCRIPTION_CHARS,
) -> str:
    """
    Keep informative description text near target length while preserving
    product attributes and quality cues.
    """
    cleaned = normalize_text(description)
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = _sentence_split(cleaned)
    if not sentences:
        return cleaned[:max_chars].rstrip(" ,;:-")

    ranked = sorted(enumerate(sentences), key=lambda x: (-_score_sentence(x[1]), x[0]))

    selected_indices: list[int] = []
    current_len = 0
    for idx, sentence in ranked:
        sentence_len = len(sentence) + (1 if selected_indices else 0)
        if current_len + sentence_len <= target_chars:
            selected_indices.append(idx)
            current_len += sentence_len

    if not selected_indices:
        selected_indices.append(ranked[0][0])

    selected_indices = sorted(set(selected_indices))
    compact = " ".join(sentences[i] for i in selected_indices)

    if len(compact) < max(180, target_chars // 2):
        for i, sentence in enumerate(sentences):
            if i in selected_indices:
                continue
            candidate = f"{compact} {sentence}".strip()
            if len(candidate) > target_chars:
                break
            compact = candidate

    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip(" ,;:-")
    return compact
