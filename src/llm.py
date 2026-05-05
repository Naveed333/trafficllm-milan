import os
import time
from openai import OpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("LLM_API_KEY") or os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get("LLM_BASE_URL")
        _client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _client


def chat(messages, model=None, max_tokens=None):
    if model is None:
        model = os.environ.get("LLM_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs = dict(model=model, messages=messages, temperature=0.0)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    for attempt in range(6):
        try:
            resp = _get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == 5:
                raise
            wait = 15 * (attempt + 1)
            print(f"  [rate limit] waiting {wait}s (attempt {attempt+1}/6) ...")
            time.sleep(wait)
