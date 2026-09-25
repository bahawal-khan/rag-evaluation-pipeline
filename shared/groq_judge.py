
import os
import re
import json as _json
import time
import asyncio
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseLLM

try:
    from json_repair import repair_json
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

MAX_RETRIES = 8
BASE_WAIT_SECONDS = 12  # Groq free tier TPM bharne mein lagta waqt


def _extract_json(text: str) -> str:
    """Response se sirf pehle '{' se aakhri '}' tak ka hissa nikal leta hai,
    taake DeepEval ka JSON parser fail na ho."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def _is_valid_json(text: str) -> bool:
    """Kabhi kabhi model malformed JSON likh deta hai (missing comma, etc.) —
    is se pehle hi validate kar lete hain taake retry ho sake."""
    try:
        _json.loads(re.sub(r",\s*([\]}])", r"\1", text))
        return True
    except Exception:
        return False


def _try_repair(text: str):
    """json-repair library se chhoti-moti formatting galtiyan (missing comma,
    unclosed bracket, etc.) khud theek karne ki koshish karta hai — is se
    dobara API call ki zaroorat nahi parti (fast + retry se zyada reliable)."""
    if not _HAS_JSON_REPAIR:
        return None
    try:
        fixed = repair_json(text)
        if fixed and fixed.strip() not in ("", "{}", "[]") and _is_valid_json(fixed):
            return fixed
    except Exception:
        pass
    return None


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.model_name = model_name
        self.model = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_retries=1,  # bada retry hum khud a_generate/generate mein handle karte hain
            reasoning_format="hidden",
            reasoning_effort="low",
            max_tokens=4096,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        for attempt in range(MAX_RETRIES):
            try:
                result = _extract_json(chat_model.invoke(prompt).content)
                if not result.strip():
                    raise ValueError("Groq se khaali response mila")
                if not _is_valid_json(result):
                    fixed = _try_repair(result)
                    if fixed:
                        return fixed
                    raise ValueError("Groq ne malformed JSON diya")
                return result
            except Exception as e:
                msg = str(e).lower()
                if "rate_limit" in msg or "429" in msg:
                    wait = BASE_WAIT_SECONDS * (attempt + 1)
                    print(f"[GroqJudge] {e} — {wait}s ruk kar retry "
                          f"kar raha hoon (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                if "khaali response" in msg or "malformed json" in msg:
                    # temperature thora barha dete hain taake retry mein wahi
                    # galat output dobara na mile (temp=0 par model deterministic hai)
                    chat_model.temperature = min(0.6, (attempt + 1) * 0.15)
                    print(f"[GroqJudge] {e} — dobara try kar raha hoon "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(2)
                    continue
                raise
        raise RuntimeError("Groq — max retries khatam ho gaye")

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        for attempt in range(MAX_RETRIES):
            try:
                res = await chat_model.ainvoke(prompt)
                result = _extract_json(res.content)
                if not result.strip():
                    raise ValueError("Groq se khaali response mila")
                if not _is_valid_json(result):
                    fixed = _try_repair(result)
                    if fixed:
                        return fixed
                    raise ValueError("Groq ne malformed JSON diya")
                return result
            except Exception as e:
                msg = str(e).lower()
                if "rate_limit" in msg or "429" in msg:
                    wait = BASE_WAIT_SECONDS * (attempt + 1)
                    print(f"[GroqJudge] {e} — {wait}s ruk kar retry "
                          f"kar raha hoon (attempt {attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                    continue
                if "khaali response" in msg or "malformed json" in msg:
                    chat_model.temperature = min(0.6, (attempt + 1) * 0.15)
                    print(f"[GroqJudge] {e} — dobara try kar raha hoon "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})...")
                    await asyncio.sleep(2)
                    continue
                raise
        raise RuntimeError("Groq — max retries khatam ho gaye")

    def get_model_name(self):
        return f"Groq ({self.model_name})"


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    judge = GroqJudge()
    print(judge.generate("Say 'GroqJudge is working' and nothing else."))