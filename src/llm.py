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


def chat(messages, model=None):
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    resp = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()
