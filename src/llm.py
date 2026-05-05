import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None

# Llama-3 chat template injected when the model has none (base model on vLLM)
_LLAMA3_TEMPLATE = (
    "{%- for message in messages %}"
        "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
        "{{- message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{%- endif %}"
)


def _get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("LLM_BASE_URL")
        if base_url:
            api_key = os.environ.get("LLM_API_KEY", "EMPTY")
            _client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def chat(messages, model=None, max_tokens=None):
    if model is None:
        model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    kwargs = dict(model=model, messages=messages, temperature=0.0)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # Inject chat template when running against a base model that has none
    if os.environ.get("LLM_INJECT_TEMPLATE", "").lower() in ("1", "true", "yes"):
        kwargs["extra_body"] = {"chat_template": _LLAMA3_TEMPLATE}

    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()
