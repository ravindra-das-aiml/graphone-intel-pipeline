"""
Startups + Products Scraper (Phase I, continued)
--------------------------------------------------
Startups: pulled from yc-oss/api, a daily-updated static mirror of Y
Combinator's own company directory (Algolia-backed on YC's site, but this
mirror needs no auth/keys and has no meaningful rate limit since it's just
static JSON on GitHub Pages). Covers 5,900+ real, funded companies.

Products: for each startup, we visit its real website and try a handful of
common pricing-page paths (/pricing, /plans, /price...). We classify
pricingModel (FREE / FREEMIUM / PAID / ENTERPRISE) using keyword heuristics
against the ACTUAL page text we fetched — never guessed. If no pricing page
is found or the site blocks us (Cloudflare etc.), pricingModel is left null
rather than invented. That null rate is itself useful signal for Phase V
(anti-bot) and is discussed in the architecture doc.

Run:
    python startups_products_scraper.py --max-companies 1000
"""

import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("startups_products_scraper")

YC_ALL_COMPANIES = "https://yc-oss.github.io/api/companies/all.json"

PRICING_PATHS = ["/pricing", "/pricing/", "/plans", "/price", "/pricing.html", "/en/pricing"]

CONCURRENCY = 40
MAX_RETRIES = 3
BASE_BACKOFF = 1.5
PAGE_TIMEOUT = 12

# Keyword heuristics — order matters, checked top to bottom, first match wins.
# All keyword checks run against the real fetched page text (lowercased).
PRICING_RULES = [
    # (label, required_all, required_any)
    ("FREEMIUM", [], ["free plan", "free tier", "start for free", "free forever"]),
    ("ENTERPRISE", [], ["contact sales", "talk to sales", "custom pricing", "request a demo"]),
    ("FREE", [], ["100% free", "completely free", "free to use", "no cost"]),
    ("PAID", [], ["/month", "/mo", "per month", "$", "subscription"]),
]


@dataclass
class StartupRecord:
    schemaVersion: str = "1.0"
    recordType: str = "STARTUP"
    source_name: str = "ycombinator.com"
    source_url: str = ""
    entityName: str = ""
    employeeCount: Optional[int] = None
    collectedAt: str = ""
    website: str = ""  # kept internally to drive the product scraper; not in schema output

    def to_schema_dict(self) -> dict:
        return {
            "schemaVersion": self.schemaVersion,
            "recordType": self.recordType,
            "source": {"name": self.source_name, "url": self.source_url},
            "content": {"entityName": self.entityName, "data": {"employeeCount": self.employeeCount}},
            "collectedAt": self.collectedAt,
        }


@dataclass
class ProductRecord:
    schemaVersion: str = "1.0"
    recordType: str = "PRODUCT"
    source_name: str = ""
    source_url: str = ""
    startupName: str = ""
    pricingModel: Optional[str] = None
    collectedAt: str = ""

    def to_schema_dict(self) -> dict:
        return {
            "schemaVersion": self.schemaVersion,
            "recordType": self.recordType,
            "source": {"name": self.source_name, "url": self.source_url},
            "content": {"startupName": self.startupName, "pricingModel": self.pricingModel},
            "collectedAt": self.collectedAt,
        }


async def fetch_with_retry(session, url, timeout=PAGE_TIMEOUT):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout),
                                    allow_redirects=True) as resp:
                if resp.status == 200:
                    return await resp.text(errors="ignore")
                if resp.status in (429, 403, 503):
                    delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(delay)
                    continue
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            await asyncio.sleep(BASE_BACKOFF * attempt)
    return None


def classify_pricing(text: str) -> Optional[str]:
    lowered = text.lower()
    for label, required_all, required_any in PRICING_RULES:
        if required_all and not all(k in lowered for k in required_all):
            continue
        if required_any and not any(k in lowered for k in required_any):
            continue
        return label
    return None


async def find_pricing_model(session, website: str) -> tuple:
    """Try each common pricing path; return (pricing_model, source_url_used)."""
    if not website:
        return None, None
    base = website.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    for path in PRICING_PATHS:
        url = base + path
        text = await fetch_with_retry(session, url)
        if text:
            model = classify_pricing(text)
            if model:
                return model, url

    text = await fetch_with_retry(session, base)
    if text:
        model = classify_pricing(text)
        if model:
            return model, base

    return None, None


async def process_company(session, sem, company: dict, now_iso: str) -> tuple:
    async with sem:
        name = company.get("name", "")
        slug = company.get("slug", "")
        website = company.get("website", "")
        yc_url = company.get("url", f"https://www.ycombinator.com/companies/{slug}")

        startup = StartupRecord(
            source_url=yc_url,
            entityName=name,
            employeeCount=company.get("team_size"),
            collectedAt=now_iso,
            website=website,
        )

        pricing_model, pricing_source = await find_pricing_model(session, website)
        product = ProductRecord(
            source_name=website or "unknown",
            source_url=pricing_source or website or yc_url,
            startupName=name,
            pricingModel=pricing_model,
            collectedAt=now_iso,
        )
        return startup, product


async def run(max_companies: int, startups_output: str, products_output: str):
    async with aiohttp.ClientSession(headers={"User-Agent": "GraphOne-Intel-Pipeline/1.0"}) as session:
        log.info("Fetching full YC company list...")
        text = await fetch_with_retry(session, YC_ALL_COMPANIES, timeout=30)
        if not text:
            log.error("Could not fetch YC company list — aborting.")
            return
        companies = json.loads(text)
        log.info(f"YC directory has {len(companies)} companies total. Using first {max_companies}.")
        companies = companies[:max_companies]

        sem = asyncio.Semaphore(CONCURRENCY)
        now_iso = datetime.now(timezone.utc).isoformat()
        tasks = [process_company(session, sem, c, now_iso) for c in companies]

        startups, products = [], []
        done = 0
        for coro in asyncio.as_completed(tasks):
            s, p = await coro
            startups.append(s)
            products.append(p)
            done += 1
            if done % 50 == 0:
                log.info(f"Processed {done}/{len(companies)}")

    with open(startups_output, "w", encoding="utf-8") as f:
        json.dump([s.to_schema_dict() for s in startups], f, indent=2, ensure_ascii=False)
    with open(products_output, "w", encoding="utf-8") as f:
        json.dump([p.to_schema_dict() for p in products], f, indent=2, ensure_ascii=False)

    with_pricing = sum(1 for p in products if p.pricingModel)
    log.info(f"Done. {len(startups)} startups -> {startups_output}")
    log.info(f"Done. {len(products)} products -> {products_output} ({with_pricing} with a detected pricing model)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-companies", type=int, default=1000)
    parser.add_argument("--startups-output", default="../../data/startups.json")
    parser.add_argument("--products-output", default="../../data/products.json")
    args = parser.parse_args()

    t0 = time.time()
    asyncio.run(run(args.max_companies, args.startups_output, args.products_output))
    log.info(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()