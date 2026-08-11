"""
News + Jobs Scraper with 24-Hour Freshness (Phase II)
--------------------------------------------------------
News: pulls from 5 real AI-focused news RSS feeds. RSS gives us a
structured, already-parsed pubDate for free (no relative-date guessing
needed for these particular sources) -- but we still ship a general
`parse_relative_date()` heuristic (handles "2 hours ago", "Yesterday",
etc.) for the Phase V/VI story: not every future source will have RSS,
and the brief explicitly asks for that fallback capability.

Jobs: pulls from 3 real public job-board APIs/feeds that expose AI/ML
roles, filtered to postings within the last 24 hours.

Full-text: for news, we fetch the actual article page and extract the
main readable text with BeautifulSoup (strip script/style/nav/footer,
keep <p> tag text) -- not just the RSS summary blurb.

IMPORTANT ABOUT FRESHNESS IN A ONE-SHOT DEMO RUN:
The brief asks for "guaranteed within the last 24 hours." In production
this pipeline would run continuously (e.g. hourly cron / distributed
workers), so the 24h window is always populated. A single one-off run
of a *demo* script may legitimately find zero qualifying items if a
source simply hasn't published anything AI-related in the last 24h at
the moment you happen to run it -- that's correct behavior, not a bug.
We do NOT relax the filter or backfill with older items to pad numbers.

Run:
    pip install beautifulsoup4
    python news_jobs_scraper.py --hours 24
"""

import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("news_jobs_scraper")

NEWS_FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("Ars Technica", "https://arstechnica.com/ai/feed/"),
    ("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
]

JOB_SOURCES = {
    "remoteok": "https://remoteok.com/api",
    "arbeitnow": "https://www.arbeitnow.com/api/job-board-api",
    "weworkremotely": "https://weworkremotely.com/categories/remote-programming-jobs.rss",
}

AI_KEYWORDS = re.compile(r"\b(ai|artificial intelligence|machine learning|ml engineer|llm|deep learning|nlp)\b", re.I)

MAX_RETRIES = 3
BASE_BACKOFF = 1.5
CONCURRENCY = 8


@dataclass
class NewsRecord:
    schemaVersion: str = "1.0"
    recordType: str = "NEWS"
    source_name: str = ""
    source_url: str = ""
    title: str = ""
    full_text: str = ""
    published_date: str = ""
    collectedAt: str = ""

    def to_schema_dict(self) -> dict:
        return {
            "schemaVersion": self.schemaVersion,
            "recordType": self.recordType,
            "source": {"name": self.source_name, "url": self.source_url},
            "content": {
                "title": self.title,
                "full_text": self.full_text,
                "published_date": self.published_date,
            },
            "collectedAt": self.collectedAt,
        }


@dataclass
class JobRecord:
    schemaVersion: str = "1.0"
    recordType: str = "JOB"
    source_name: str = ""
    source_url: str = ""
    company: str = ""
    date: str = ""
    is_remote: bool = True
    role_family: str = "Engineering"
    title: str = ""
    collectedAt: str = ""

    def to_schema_dict(self) -> dict:
        return {
            "schemaVersion": self.schemaVersion,
            "recordType": self.recordType,
            "source": {"name": self.source_name, "url": self.source_url},
            "content": {
                "company": self.company,
                "date": self.date,
                "is_remote": self.is_remote,
                "role_family": self.role_family,
                "title": self.title,
            },
            "collectedAt": self.collectedAt,
        }


def parse_relative_date(text: str, now: datetime) -> Optional[datetime]:
    """
    Fallback normalizer for sources that only give relative timestamps
    instead of a real date (e.g. "2 hours ago", "Yesterday", "3d ago").
    Only used when a source has no structured date field to parse.
    """
    text = text.strip().lower()
    if "today" in text:
        return now
    if "yesterday" in text:
        return now - timedelta(days=1)
    m = re.match(r"(\d+)\s*(second|minute|hour|day|week)s?\s*ago", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        unit_map = {"second": "seconds", "minute": "minutes", "hour": "hours",
                    "day": "days", "week": "weeks"}
        return now - timedelta(**{unit_map[unit]: n})
    m = re.match(r"(\d+)([smhdw])\s*ago", text)  # short form "3h ago", "2d ago"
    if m:
        n, unit = int(m.group(1)), m.group(2)
        unit_map = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
        return now - timedelta(**{unit_map[unit]: n})
    return None


async def fetch_with_retry(session, url, timeout=15) -> Optional[str]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    return await resp.text(errors="ignore")
                if resp.status in (429, 403, 503):
                    delay = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(f"{resp.status} on {url} — retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await asyncio.sleep(BASE_BACKOFF * attempt)
    return None


def extract_full_text(html: str) -> str:
    """Strip boilerplate and return the readable article text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    article = soup.find("article") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)  # drop short boilerplate lines
    return text[:20000]  # cap length; Phase III chunking handles the rest


async def process_news_entry(session, sem, source_name, entry, cutoff, now):
    async with sem:
        pub_dt = None
        if getattr(entry, "published_parsed", None):
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        elif getattr(entry, "updated_parsed", None):
            pub_dt = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        else:
            pub_dt = parse_relative_date(getattr(entry, "published", ""), now)

        if not pub_dt or pub_dt < cutoff:
            return None  # not within freshness window — skip, don't fabricate

        link = entry.get("link", "")
        html = await fetch_with_retry(session, link) if link else None
        full_text = extract_full_text(html) if html else entry.get("summary", "")

        return NewsRecord(
            source_name=source_name,
            source_url=link,
            title=entry.get("title", "").strip(),
            full_text=full_text,
            published_date=pub_dt.isoformat(),
            collectedAt=now.isoformat(),
        )


async def scrape_news(session, sem, hours, now):
    cutoff = now - timedelta(hours=hours)
    all_news = []
    for source_name, feed_url in NEWS_FEEDS:
        log.info(f"Fetching news feed: {source_name}")
        xml_text = await fetch_with_retry(session, feed_url)
        if not xml_text:
            log.warning(f"Could not fetch {source_name}, skipping.")
            continue
        feed = feedparser.parse(xml_text)
        tasks = [process_news_entry(session, sem, source_name, e, cutoff, now) for e in feed.entries]
        results = await asyncio.gather(*tasks)
        fresh = [r for r in results if r]
        log.info(f"{source_name}: {len(fresh)} articles within last {hours}h (out of {len(feed.entries)} in feed)")
        all_news.extend(fresh)
    return all_news


async def scrape_remoteok(session, cutoff, now):
    text = await fetch_with_retry(session, JOB_SOURCES["remoteok"])
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    jobs = []
    for item in data:
        if not isinstance(item, dict) or "date" not in item:
            continue  # first element is often metadata, not a job
        title = item.get("position", "") or item.get("title", "")
        desc = item.get("description", "")
        if not AI_KEYWORDS.search(title + " " + desc):
            continue
        try:
            posted = datetime.fromisoformat(item["date"].replace("Z", "+00:00"))
        except (ValueError, KeyError):
            continue
        if posted < cutoff:
            continue
        jobs.append(JobRecord(
            source_name="RemoteOK",
            source_url=item.get("url", ""),
            company=item.get("company", ""),
            date=posted.isoformat(),
            is_remote=True,
            role_family="Engineering",
            title=title,
            collectedAt=now.isoformat(),
        ))
    return jobs


async def scrape_arbeitnow(session, cutoff, now):
    text = await fetch_with_retry(session, JOB_SOURCES["arbeitnow"])
    if not text:
        return []
    try:
        data = json.loads(text).get("data", [])
    except json.JSONDecodeError:
        return []
    jobs = []
    for item in data:
        title = item.get("title", "")
        desc = item.get("description", "")
        if not AI_KEYWORDS.search(title + " " + desc):
            continue
        ts = item.get("created_at")
        if not ts:
            continue
        try:
            posted = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (ValueError, TypeError):
            continue
        if posted < cutoff:
            continue
        jobs.append(JobRecord(
            source_name="Arbeitnow",
            source_url=item.get("url", ""),
            company=item.get("company_name", ""),
            date=posted.isoformat(),
            is_remote=item.get("remote", True),
            role_family="Engineering",
            title=title,
            collectedAt=now.isoformat(),
        ))
    return jobs


async def scrape_wwr(session, cutoff, now):
    text = await fetch_with_retry(session, JOB_SOURCES["weworkremotely"])
    if not text:
        return []
    feed = feedparser.parse(text)
    jobs = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        if not AI_KEYWORDS.search(title + " " + summary):
            continue
        pub_dt = None
        if getattr(entry, "published_parsed", None):
            pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if not pub_dt or pub_dt < cutoff:
            continue
        company = title.split(":")[0].strip() if ":" in title else ""
        jobs.append(JobRecord(
            source_name="We Work Remotely",
            source_url=entry.get("link", ""),
            company=company,
            date=pub_dt.isoformat(),
            is_remote=True,
            role_family="Engineering",
            title=title,
            collectedAt=now.isoformat(),
        ))
    return jobs


async def run(hours: int, news_output: str, jobs_output: str):
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)
    sem = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(headers={"User-Agent": "GraphOne-Intel-Pipeline/1.0"}) as session:
        news = await scrape_news(session, sem, hours, now)

        log.info("Fetching job boards...")
        remoteok_jobs = await scrape_remoteok(session, cutoff, now)
        arbeitnow_jobs = await scrape_arbeitnow(session, cutoff, now)
        wwr_jobs = await scrape_wwr(session, cutoff, now)
        jobs = remoteok_jobs + arbeitnow_jobs + wwr_jobs
        log.info(f"Jobs found within {hours}h: RemoteOK={len(remoteok_jobs)}, "
                 f"Arbeitnow={len(arbeitnow_jobs)}, WWR={len(wwr_jobs)}")

    with open(news_output, "w", encoding="utf-8") as f:
        json.dump([n.to_schema_dict() for n in news], f, indent=2, ensure_ascii=False)
    with open(jobs_output, "w", encoding="utf-8") as f:
        json.dump([j.to_schema_dict() for j in jobs], f, indent=2, ensure_ascii=False)

    log.info(f"Done. {len(news)} news articles -> {news_output}")
    log.info(f"Done. {len(jobs)} jobs -> {jobs_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24, help="Freshness window in hours")
    parser.add_argument("--news-output", default="../../data/news.json")
    parser.add_argument("--jobs-output", default="../../data/jobs.json")
    args = parser.parse_args()

    t0 = time.time()
    asyncio.run(run(args.hours, args.news_output, args.jobs_output))
    log.info(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()