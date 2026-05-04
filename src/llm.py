import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def chat(messages, model=None, max_tokens=None):
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = dict(model=model, messages=messages, temperature=0.0)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()
