import base64
import io
from typing import Any, Dict, Literal
from urllib.parse import urlparse


ProtocolType = Literal["openai", "anthropic"]


def resolve_api_endpoint(base_url: str, protocol: ProtocolType) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("api_base_url is required")

    parsed = urlparse(normalized)
    path = parsed.path.rstrip("/")
    if protocol == "openai":
        if path.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"
    if path.endswith("/messages"):
        return normalized
    return f"{normalized}/messages"


def encode_image_to_jpeg_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_openai_payload(
    model_name: str,
    prompt: str,
    image_data_url: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "max_tokens": max_new_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
    }


def build_anthropic_payload(
    model_name: str,
    prompt: str,
    image_base64: str,
    media_type: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    return {
        "model": model_name,
        "max_tokens": max_new_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def parse_openai_caption(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("OpenAI response missing choices")

    content = ((choices[0] or {}).get("message") or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            item.get("text", "").strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        merged = "\n".join(part for part in text_parts if part)
        if merged:
            return merged
    raise ValueError("OpenAI response did not contain usable text content")


def parse_anthropic_caption(response_json: Dict[str, Any]) -> str:
    content = response_json.get("content") or []
    text_parts = [
        item.get("text", "").strip()
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    merged = "\n".join(part for part in text_parts if part)
    if not merged:
        raise ValueError("Anthropic response did not contain usable text content")
    return merged
