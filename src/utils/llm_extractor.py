"""
Multi-Tier LLM Extraction Engine (Phase III)
-----------------------------------------------
A reusable module that turns raw text into structured JSON using an LLM,
with a resilient fallback chain: Gemini Flash -> Groq (Llama 3) -> DeepSeek.

Why a fallback chain instead of one provider:
- Any single provider can rate-limit us (429), have an outage (5xx), or
  reject an oversized payload (413). At the scale the brief describes
  (thousands of concurrent extractions), relying on one provider is a
  single point of failure. If Gemini is exhausted or down, we don't
  stall the pipeline -- we fall through to Groq, then DeepSeek.

How each failure mode is handled:
- 429 (rate limited): exponential backoff + jitter, retried a few times
  on the SAME provider first (cheap, fast recovery for transient limits)
  before giving up on that provider and falling through to the next.
- 413 / "context length" errors: instead of retrying (which would just
  fail again), we split the input into smaller chunks via chunk_text()
  and extract each chunk separately, then merge the results.
- 5xx / network errors: same backoff+retry, then fallback.
- Total failure on all three providers: the function returns None rather
  than inventing a fake extraction -- callers must treat that record as
  "unstructured, needs reprocessing" rather than fabricate data.

Run standalone for a quick manual test:
    python llm_extractor.py --text "OpenAI announced GPT-5 today, and Google DeepMind responded within hours."
"""

import argparse
import asyncio
import json
import logging
import os
import random
import re
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("llm_extractor")

MAX_RETRIES_PER_PROVIDER = 3
BASE_BACKOFF = 2.0
MAX_CHUNK_CHARS = 6000  # conservative -- keeps well clear of any provider's context limit


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """
    Split text into chunks that stay under max_chars, breaking on paragraph
    boundaries where possible so we don't cut a sentence in half and lose
    semantic meaning right at the chunk edge.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i:i + max_chars])
                current = ""
            else:
                current = para + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _extract_json(raw: str) -> Optional[dict]:
    """LLMs often wrap JSON in ```json fences or add stray text -- clean it up."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


async def _call_gemini(session: aiohttp.ClientSession, prompt: str) -> tuple:
    """Returns (result_dict_or_None, should_fallback: bool)."""
    if not GEMINI_API_KEY:
        return None, True
    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1}}

    for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
        try:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return _extract_json(text), False
                if resp.status == 429:
                    delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"Gemini 429 — retrying in {delay:.1f}s (attempt {attempt})")
                    await asyncio.sleep(delay)
                    continue
                if resp.status == 413:
                    log.warning("Gemini 413 (payload too large) — signaling caller to chunk smaller")
                    return None, "chunk"
                if resp.status in (500, 502, 503):
                    delay = BASE_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                log.warning(f"Gemini unexpected status {resp.status} — falling back")
                return None, True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(BASE_BACKOFF * attempt)
    log.warning("Gemini exhausted retries — falling back to Groq")
    return None, True


async def _call_openai_compatible(session: aiohttp.ClientSession, url: str, api_key: str,
                                   model: str, prompt: str, provider_name: str) -> tuple:
    """Shared logic for Groq and DeepSeek, both OpenAI-compatible chat APIs."""
    if not api_key:
        return None, True
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }

    for attempt in range(1, MAX_RETRIES_PER_PROVIDER + 1):
        try:
            async with session.post(url, json=body, headers=headers,
                                     timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
                    return _extract_json(text), False
                if resp.status == 429:
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after else BASE_BACKOFF * (2 ** attempt)
                    delay += random.uniform(0, 1)
                    log.warning(f"{provider_name} 429 — retrying in {delay:.1f}s (attempt {attempt})")
                    await asyncio.sleep(delay)
                    continue
                if resp.status == 413:
                    log.warning(f"{provider_name} 413 (payload too large) — signaling caller to chunk smaller")
                    return None, "chunk"
                if resp.status in (500, 502, 503):
                    await asyncio.sleep(BASE_BACKOFF * (2 ** attempt))
                    continue
                log.warning(f"{provider_name} unexpected status {resp.status} — falling back")
                return None, True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(BASE_BACKOFF * attempt)
    log.warning(f"{provider_name} exhausted retries — falling back")
    return None, True


async def extract_structured(session: aiohttp.ClientSession, prompt: str) -> Optional[dict]:
    """
    Try Gemini -> Groq -> DeepSeek in order. If a provider signals the
    payload was too large (413), split the prompt's text portion into
    chunks and merge per-chunk results rather than failing outright.
    """
    result, fallback_signal = await _call_gemini(session, prompt)
    if result is not None:
        return result
    if fallback_signal == "chunk":
        return await _extract_chunked(session, prompt)

    result, fallback_signal = await _call_openai_compatible(
        session, GROQ_URL, GROQ_API_KEY, "llama-3.1-8b-instant", prompt, "Groq")
    if result is not None:
        return result
    if fallback_signal == "chunk":
        return await _extract_chunked(session, prompt)

    result, fallback_signal = await _call_openai_compatible(
        session, DEEPSEEK_URL, DEEPSEEK_API_KEY, "deepseek-chat", prompt, "DeepSeek")
    if result is not None:
        return result

    log.error("All three providers failed for this prompt.")
    return None


async def _extract_chunked(session: aiohttp.ClientSession, original_prompt: str) -> Optional[dict]:
    """Fallback path when a provider rejects the payload as too large."""
    chunks = chunk_text(original_prompt, max_chars=MAX_CHUNK_CHARS // 2)
    merged_mentions = []
    for chunk in chunks:
        result = await extract_structured(session, chunk)
        if result and "companies_mentioned" in result:
            merged_mentions.extend(result["companies_mentioned"])
    if not merged_mentions:
        return None
    return {"companies_mentioned": list(dict.fromkeys(merged_mentions))}


def build_news_extraction_prompt(title: str, full_text: str) -> str:
    """
    Prompt used to turn a raw news article into structured signal.
    Kept intentionally narrow (a few well-defined fields) since precise,
    schema-bound extraction is far more reliable than open-ended summarization.
    """
    text = full_text[:MAX_CHUNK_CHARS]
    return f"""You are a data extraction engine. Given a news article, return ONLY a JSON object
(no prose, no markdown fences) with exactly these fields:
- "companies_mentioned": array of distinct company/startup names mentioned in the article
- "topic": one of ["funding", "product_launch", "research", "policy", "acquisition", "other"]
- "sentiment": one of ["positive", "neutral", "negative"]

Article title: {title}
Article text: {text}

Return only the JSON object."""


async def enrich_news_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    async with aiohttp.ClientSession() as session:
        for i, article in enumerate(articles):
            title = article["content"]["title"]
            full_text = article["content"]["full_text"]
            prompt = build_news_extraction_prompt(title, full_text)
            result = await extract_structured(session, prompt)
            article["content"]["llm_extraction"] = result
            log.info(f"[{i+1}/{len(articles)}] {title[:60]!r} -> {result}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    log.info(f"Done. Enriched {len(articles)} articles -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Quick manual test: extract from a single string")
    parser.add_argument("--news-input", default="../../data/news.json")
    parser.add_argument("--news-output", default="../../data/news_enriched.json")
    parser.add_argument("--mode", choices=["test", "enrich-news"], default="enrich-news")
    args = parser.parse_args()

    async def _main():
        async with aiohttp.ClientSession() as session:
            if args.mode == "test" or args.text:
                prompt = build_news_extraction_prompt("Test Article", args.text or "OpenAI announced GPT-5 today.")
                result = await extract_structured(session, prompt)
                print(json.dumps(result, indent=2))
            else:
                await enrich_news_file(args.news_input, args.news_output)

    asyncio.run(_main())


if __name__ == "__main__":
    main()