"""
Entity Resolution Engine (Phase IV)
--------------------------------------
Canonicalizes messy startup/product/company name strings against a seed
list of known AI companies, e.g. "OpenAI", "OpenAI, Inc.", "Open AI" all
resolve to the canonical "OpenAI".

Two-stage matching, cheapest/most-precise first:
1. Normalized exact match: strip legal suffixes (Inc., LLC, Ltd...),
   punctuation, and casing, then compare directly. Fast, zero false
   positives.
2. Fuzzy match (difflib, stdlib -- no extra dependency): only used if
   stage 1 finds nothing, and only accepted above a similarity cutoff.
   Every fuzzy match is logged with its confidence score so a human can
   audit borderline calls -- we don't silently auto-merge uncertain pairs.

Anything that matches neither stage is left unmatched (raw name kept
as-is) rather than forced into the nearest canonical entity -- a wrong
merge is worse than no merge, and the brief penalizes fabricated
precision, not honest "no match" results.

Run:
    python entity_resolver.py
"""

import argparse
import difflib
import json
import logging
import re
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("entity_resolver")

CANONICAL_STARTUPS = [
    "OpenAI", "Anthropic", "Google DeepMind", "Mistral AI", "Cohere",
    "Stability AI", "Hugging Face", "Perplexity AI", "Scale AI", "Databricks",
    "xAI", "Inflection AI", "Character.AI", "Runway", "Adept AI",
    "Together AI", "Replit", "LangChain", "Weights & Biases", "Pinecone",
    "Modal", "Fireworks AI", "Groq", "Cerebras", "SambaNova",
    "Glean", "Harvey", "Sierra", "Anysphere", "Suno",
    "ElevenLabs", "Synthesia", "Midjourney", "Luma AI", "Pika",
    "Tempus AI", "Abridge", "Hippocratic AI", "Waymo", "Cruise",
    "Aurora Innovation", "Figure AI", "Physical Intelligence", "Skild AI", "World Labs",
    "Sakana AI", "Reka AI", "AI21 Labs", "01.AI", "Baseten",
]

SUFFIX_PATTERN = re.compile(
    r"[,.\s]*\b(inc\.?|incorporated|llc|ltd\.?|limited|corp\.?|corporation|"
    r"co\.?|plc|gmbh|pte\.?|pbc|technologies|technology)\b\.?\s*$",
    re.IGNORECASE,
)

FUZZY_CUTOFF = 0.84


def normalize(name: str) -> str:
    if not name:
        return ""
    n = name.strip()
    prev = None
    while prev != n:
        prev = n
        n = SUFFIX_PATTERN.sub("", n).strip()
    n = re.sub(r"[^\w\s]", "", n)
    n = re.sub(r"\s+", " ", n).strip().lower()
    return n


def build_normalized_index(canonical_list):
    return {normalize(c): c for c in canonical_list}


def resolve_name(raw_name: str, canonical_list, normalized_index) -> dict:
    normalized_raw = normalize(raw_name)
    if not normalized_raw:
        return {"raw_name": raw_name, "canonical_name": None, "method": "empty", "confidence": 0.0}

    if normalized_raw in normalized_index:
        return {
            "raw_name": raw_name,
            "canonical_name": normalized_index[normalized_raw],
            "method": "exact_normalized",
            "confidence": 1.0,
        }

    candidates = list(normalized_index.keys())
    matches = difflib.get_close_matches(normalized_raw, candidates, n=1, cutoff=FUZZY_CUTOFF)
    if matches:
        score = difflib.SequenceMatcher(None, normalized_raw, matches[0]).ratio()
        return {
            "raw_name": raw_name,
            "canonical_name": normalized_index[matches[0]],
            "method": "fuzzy",
            "confidence": round(score, 3),
        }

    return {"raw_name": raw_name, "canonical_name": None, "method": "unmatched", "confidence": 0.0}


def collect_raw_names(startups_path, products_path, news_enriched_path):
    raw_names = []

    try:
        with open(startups_path, encoding="utf-8") as f:
            for s in json.load(f):
                name = s.get("content", {}).get("entityName")
                if name:
                    raw_names.append((name, "startups.json"))
    except FileNotFoundError:
        log.warning(f"{startups_path} not found, skipping.")

    try:
        with open(products_path, encoding="utf-8") as f:
            for p in json.load(f):
                name = p.get("content", {}).get("startupName")
                if name:
                    raw_names.append((name, "products.json"))
    except FileNotFoundError:
        log.warning(f"{products_path} not found, skipping.")

    try:
        with open(news_enriched_path, encoding="utf-8") as f:
            for article in json.load(f):
                extraction = article.get("content", {}).get("llm_extraction") or {}
                for company in extraction.get("companies_mentioned", []):
                    raw_names.append((company, "news_enriched.json"))
    except FileNotFoundError:
        log.warning(f"{news_enriched_path} not found, skipping.")

    return raw_names


def run(startups_path, products_path, news_enriched_path, output_path):
    normalized_index = build_normalized_index(CANONICAL_STARTUPS)
    raw_entries = collect_raw_names(startups_path, products_path, news_enriched_path)
    log.info(f"Collected {len(raw_entries)} raw entity mentions to resolve.")

    seen = {}
    log_rows = []
    for raw_name, source in raw_entries:
        key = (raw_name, source)
        if key in seen:
            continue
        seen[key] = True
        result = resolve_name(raw_name, CANONICAL_STARTUPS, normalized_index)
        result["source"] = source
        result["resolvedAt"] = datetime.now(timezone.utc).isoformat()
        log_rows.append(result)

    matched = sum(1 for r in log_rows if r["canonical_name"])
    log.info(f"Resolved {matched}/{len(log_rows)} unique raw names to a canonical entity "
              f"({matched/len(log_rows)*100:.1f}% match rate)." if log_rows else "No raw names found.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log_rows, f, indent=2, ensure_ascii=False)
    log.info(f"Done. Entity mapping log -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--startups", default="../../data/startups.json")
    parser.add_argument("--products", default="../../data/products.json")
    parser.add_argument("--news-enriched", default="../../data/news_enriched.json")
    parser.add_argument("--output", default="../../data/entity_mapping_log.json")
    args = parser.parse_args()
    run(args.startups, args.products, args.news_enriched, args.output)


if __name__ == "__main__":
    main()