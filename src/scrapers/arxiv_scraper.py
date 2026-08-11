"""
Arxiv Research Paper Scraper (Phase I)
----------------------------------------
Fetches AI research papers from the Arxiv API, then tries to find an
associated GitHub repository and fetches the CURRENT star count for
that repo via the GitHub REST API.

Output: JSON records matching the "Research Paper Entity" schema from
the task doc (schemaVersion, recordType, content.title, authors,
paper_url, github_url, github_stars, published_date).

Design notes (why it's built this way):
- Uses asyncio + aiohttp so hundreds/thousands of paper lookups run
  concurrently instead of one-by-one (needed to scale toward 500k+
  eventually — you'd just raise CONCURRENCY / add more worker nodes,
  no code change).
- A semaphore caps concurrency so we don't get rate-limited by Arxiv.
  GitHub's Search API has its own, much stricter limit, so it gets its
  own dedicated pacing (see GITHUB_SEARCH_SEM below).
- Every network call is wrapped in retry-with-exponential-backoff +
  jitter, because Phase III/V of the brief explicitly call this out.
- Nothing here is invented: if we can't find a real GitHub repo for a
  paper, github_url/github_stars are left null rather than guessed —
  the brief says hallucinated data = disqualification.

Run:
    pip install aiohttp feedparser python-dotenv
    python arxiv_scraper.py --query "cat:cs.AI" --max-results 1000
"""

import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import os

import aiohttp
import feedparser
from dotenv import load_dotenv

load_dotenv()  # reads GITHUB_TOKEN etc. from the .env file in project root
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("arxiv_scraper")

ARXIV_API = "http://export.arxiv.org/api/query"
# NOTE: Papers with Code was shut down by Meta in July 2025 (site now
# redirects to Hugging Face, its API returns nothing). We use GitHub's
# own Search API instead: it lets us search repos whose README/description
# mentions the paper's Arxiv ID, and the search response already includes
# each repo's live star count — one call does both jobs.
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
GITHUB_API = "https://api.github.com/repos/"

CONCURRENCY = 8            # simultaneous in-flight requests (raise for scale)
MAX_RETRIES = 5
BASE_BACKOFF = 1.5         # seconds, doubles each retry + jitter

# GitHub's Search API allows ~30 requests/minute even with a token — much
# stricter than its normal REST endpoints. We pace search calls through
# their own semaphore + a small delay so we mostly avoid 403s instead of
# constantly hitting them and paying for it in backoff sleeps.
GITHUB_SEARCH_SEM = asyncio.Semaphore(1)
GITHUB_SEARCH_MIN_INTERVAL = 2.1  # seconds between search calls (~28/min)


@dataclass
class ResearchPaperRecord:
    schemaVersion: str = "1.0"
    recordType: str = "RESEARCH_PAPER"
    source_name: str = "arxiv.org"
    source_url: str = ""
    title: str = ""
    authors: list = field(default_factory=list)
    paper_url: str = ""
    github_url: Optional[str] = None
    github_stars: Optional[int] = None
    published_date: str = ""
    collectedAt: str = ""

    def to_schema_dict(self) -> dict:
        """Nest fields the way the brief's schema table expects."""
        return {
            "schemaVersion": self.schemaVersion,
            "recordType": self.recordType,
            "source": {"name": self.source_name, "url": self.source_url},
            "content": {
                "title": self.title,
                "authors": self.authors,
                "paper_url": self.paper_url,
                "github_url": self.github_url,
                "github_stars": self.github_stars,
                "published_date": self.published_date,
            },
            "collectedAt": self.collectedAt,
        }


async def fetch_with_retry(session: aiohttp.ClientSession, url: str,
                            headers: Optional[dict] = None,
                            params: Optional[dict] = None) -> Optional[str]:
    """GET a URL with exponential backoff + jitter on 429 / 403 / 5xx / timeouts."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, headers=headers, params=params,
                                    timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status == 200:
                    return await resp.text()
                if resp.status == 429:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else BASE_BACKOFF * (2 ** attempt)
                    delay += random.uniform(0, 1)  # jitter
                    log.warning(f"429 rate limited on {url} — sleeping {delay:.1f}s (attempt {attempt})")
                    await asyncio.sleep(delay)
                    continue
                if resp.status == 403:
                    # GitHub's Search API returns 403 (not 429) when its
                    # separate, stricter rate limit is hit. Same backoff.
                    reset_header = resp.headers.get("X-RateLimit-Reset")
                    if reset_header:
                        wait_secs = max(float(reset_header) - time.time(), 1)
                        delay = min(wait_secs, 60) + random.uniform(0, 1)
                    else:
                        delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"403 (likely rate limit) on {url} — sleeping {delay:.1f}s (attempt {attempt})")
                    await asyncio.sleep(delay)
                    continue
                if resp.status in (500, 502, 503, 504):
                    delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"{resp.status} on {url} — retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                if resp.status == 404:
                    return None  # not found, no point retrying
                log.warning(f"Unexpected status {resp.status} on {url}")
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            log.warning(f"Network error on {url}: {e} — retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    log.error(f"Giving up on {url} after {MAX_RETRIES} attempts")
    return None


def extract_arxiv_id(paper_url: str) -> Optional[str]:
    m = re.search(r"abs/([\w.\-/]+)", paper_url)
    return m.group(1) if m else None


async def find_github_repo_and_stars(session: aiohttp.ClientSession, arxiv_id: str,
                                      headers: Optional[dict] = None) -> tuple:
    """
    Search GitHub itself for a repo that references this paper's Arxiv ID
    in its README (the standard way ML authors link code to a paper).
    Returns (repo_url, stars) or (None, None) if nothing plausible is found.
    One API call gives us both the repo and its live star count.
    """
    bare_id = re.sub(r"v\d+$", "", arxiv_id)
    params = {
        "q": f'"{bare_id}" in:readme',
        "sort": "stars",
        "order": "desc",
        "per_page": 1,
    }
    async with GITHUB_SEARCH_SEM:
        text = await fetch_with_retry(session, GITHUB_SEARCH_API, headers=headers, params=params)
        await asyncio.sleep(GITHUB_SEARCH_MIN_INTERVAL)
    if not text:
        return None, None
    try:
        data = json.loads(text)
        items = data.get("items", [])
        if items:
            top = items[0]
            return top.get("html_url"), top.get("stargazers_count")
    except json.JSONDecodeError:
        return None, None
    return None, None


async def process_entry(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                         entry) -> ResearchPaperRecord:
    async with sem:
        paper_url = entry.get("id", "")
        arxiv_id = extract_arxiv_id(paper_url)

        record = ResearchPaperRecord(
            source_url=paper_url,
            title=entry.get("title", "").strip().replace("\n", " "),
            authors=[a.get("name") for a in entry.get("authors", [])],
            paper_url=paper_url,
            published_date=entry.get("published", ""),
            collectedAt=datetime.now(timezone.utc).isoformat(),
        )

        if arxiv_id:
            gh_headers = {"Accept": "application/vnd.github+json"}
            if GITHUB_TOKEN:
                gh_headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            repo_url, stars = await find_github_repo_and_stars(session, arxiv_id, headers=gh_headers)
            record.github_url = repo_url
            record.github_stars = stars

        return record


async def run(query: str, max_results: int, batch_size: int, output_path: str):
    sem = asyncio.Semaphore(CONCURRENCY)
    all_records = []

    async with aiohttp.ClientSession(headers={"User-Agent": "GraphOne-Intel-Pipeline/1.0"}) as session:
        start = 0
        while start < max_results:
            page_size = min(batch_size, max_results - start)
            params = {
                "search_query": query,
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            log.info(f"Fetching Arxiv results {start}..{start + page_size}")
            xml_text = await fetch_with_retry(session, ARXIV_API, params=params)
            if not xml_text:
                log.error("Failed to fetch Arxiv page, stopping.")
                break

            feed = feedparser.parse(xml_text)
            if not feed.entries:
                log.info("No more entries returned by Arxiv — stopping early.")
                break

            tasks = [process_entry(session, sem, e) for e in feed.entries]
            results = await asyncio.gather(*tasks)
            all_records.extend(results)

            log.info(f"Total collected so far: {len(all_records)}")
            start += page_size
            await asyncio.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.to_schema_dict() for r in all_records], f, indent=2, ensure_ascii=False)

    log.info(f"Done. Wrote {len(all_records)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Arxiv + GitHub stars scraper")
    parser.add_argument("--query", default="cat:cs.AI", help="Arxiv search_query, e.g. 'cat:cs.AI' or 'all:transformers'")
    parser.add_argument("--max-results", type=int, default=100, help="Total papers to fetch")
    parser.add_argument("--batch-size", type=int, default=50, help="Papers per Arxiv API page (Arxiv caps this)")
    parser.add_argument("--output", default="../../data/research_papers.json")
    args = parser.parse_args()

    t0 = time.time()
    asyncio.run(run(args.query, args.max_results, args.batch_size, args.output))
    log.info(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()