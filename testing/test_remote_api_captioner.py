import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extensions_built_in.captioner.remote_api_utils import (
    build_anthropic_payload,
    build_openai_payload,
    parse_anthropic_caption,
    parse_openai_caption,
    resolve_api_endpoint,
)


def test_openai_payload_contains_model_prompt_and_image():
    payload = build_openai_payload(
        model_name="gpt-4.1-mini",
        prompt="Describe this image.",
        image_data_url="data:image/jpeg;base64,abc123",
        max_new_tokens=256,
    )

    assert payload["model"] == "gpt-4.1-mini"
    assert payload["max_tokens"] == 256
    assert payload["messages"][0]["content"][0]["text"] == "Describe this image."
    assert payload["messages"][0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,abc123"


def test_anthropic_payload_contains_model_prompt_and_image():
    payload = build_anthropic_payload(
        model_name="claude-3-5-sonnet-latest",
        prompt="Describe this image.",
        image_base64="abc123",
        media_type="image/jpeg",
        max_new_tokens=256,
    )

    assert payload["model"] == "claude-3-5-sonnet-latest"
    assert payload["max_tokens"] == 256
    assert payload["messages"][0]["content"][0]["type"] == "image"
    assert payload["messages"][0]["content"][0]["source"]["data"] == "abc123"
    assert payload["messages"][0]["content"][1]["text"] == "Describe this image."


def test_parse_openai_caption_handles_string_and_array_content():
    string_response = {
        "choices": [
            {
                "message": {
                    "content": "A black dragon flying above snowy mountains."
                }
            }
        ]
    }
    array_response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "A black dragon flying above snowy mountains."}
                    ]
                }
            }
        ]
    }

    assert parse_openai_caption(string_response) == "A black dragon flying above snowy mountains."
    assert parse_openai_caption(array_response) == "A black dragon flying above snowy mountains."


def test_parse_anthropic_caption_reads_text_blocks():
    response = {
        "content": [
            {"type": "text", "text": "A black dragon flying above snowy mountains."}
        ]
    }

    assert parse_anthropic_caption(response) == "A black dragon flying above snowy mountains."


def test_resolve_api_endpoint_appends_protocol_path_when_needed():
    assert resolve_api_endpoint("https://api.example.com/v1", "openai") == "https://api.example.com/v1/chat/completions"
    assert resolve_api_endpoint("https://api.example.com/v1/chat/completions", "openai") == "https://api.example.com/v1/chat/completions"
    assert resolve_api_endpoint("https://api.example.com", "anthropic") == "https://api.example.com/messages"
    assert resolve_api_endpoint("https://api.example.com/messages", "anthropic") == "https://api.example.com/messages"


if __name__ == "__main__":
    test_openai_payload_contains_model_prompt_and_image()
    test_anthropic_payload_contains_model_prompt_and_image()
    test_parse_openai_caption_handles_string_and_array_content()
    test_parse_anthropic_caption_reads_text_blocks()
    test_resolve_api_endpoint_appends_protocol_path_when_needed()
