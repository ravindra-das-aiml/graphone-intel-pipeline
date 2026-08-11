# GraphOne / FrontierAtlas — Intelligence Graph Pipeline (AI Engineer Demo Task)

All 6 phases of the brief are implemented and run against real, verifiable
sources — no fabricated/hallucinated records anywhere in the outputs.

**Google Sheet (6 tabs, live data):** [link here]

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your own API keys
```

## Phase I — Research Papers, Startups & Products

```bash
cd src/scrapers
python arxiv_scraper.py --query "cat:cs.AI" --max-results 1000
python startups_products_scraper.py --max-companies 1000
```

- **Papers**: pulled from the Arxiv API. For each paper, we search GitHub's
  own Search API for a repo referencing the paper's Arxiv ID in its README,
  and pull that repo's live star count in the same call. (Papers with Code
  was shut down by Meta in July 2025, so we don't depend on it.)
- **Startups**: pulled from `yc-oss/api`, a daily-updated static mirror of
  Y Combinator's company directory — 1,000 real, funded companies.
- **Products**: for each startup, we fetch its real website's pricing page
  (`/pricing`, `/plans`, etc.) and classify `pricingModel` from keyword
  heuristics against the *actual fetched text* — never guessed. If no
  pricing signal is found, the field is left null rather than invented.
- All three scrapers are async (aiohttp), with a concurrency semaphore and
  exponential-backoff-with-jitter retries on 429/403/5xx. To scale toward
  500k+: raise concurrency, shard the source (Arxiv categories / YC
  batches), and swap the JSON file sink for a queue → database. No logic
  changes needed, only infrastructure — see `architecture.pdf`.

## Phase II — News + Jobs (24h freshness)

```bash
python news_jobs_scraper.py --hours 24
```

- 5 real AI news RSS feeds (TechCrunch, VentureBeat, The Verge, Ars
  Technica, MIT Tech Review) — full article text extracted with
  BeautifulSoup, not just the RSS summary.
- 3 real job-board APIs/feeds (RemoteOK, Arbeitnow, We Work Remotely),
  filtered to AI/ML roles posted in the last 24 hours.
- Includes a `parse_relative_date()` fallback for sources that only give
  relative timestamps ("2 hours ago") instead of structured dates.
- Freshness filtering is strict: if a source has nothing genuinely new in
  the window, we return zero rather than backfilling with older items.

## Phase III — Multi-Tier LLM Extraction Engine

```bash
cd ../utils
python llm_extractor.py --news-input ../../data/news.json --news-output ../../data/news_enriched.json
```

- Fallback chain: **Gemini Flash → Groq (Llama 3) → DeepSeek**.
- 429s get exponential backoff + jitter, retried on the same provider
  first, then fall through to the next provider.
- 413s trigger `chunk_text()` — the payload is split into smaller pieces
  and re-extracted, rather than retried as-is.
- All three providers failing returns `None`, not a fabricated result.

## Phase IV — Entity Resolution

```bash
python entity_resolver.py
```

- Two-stage matching against a 50-company canonical seed list: (1)
  normalized exact match after stripping legal suffixes (Inc., LLC, PBC...),
  (2) fuzzy match (stdlib `difflib`) above a conservative similarity cutoff.
- Every match is logged with its method and confidence score. Unmatched
  names are left as-is rather than force-merged.
- Verified against the brief's own example: `"OpenAI"`, `"OpenAI, Inc."`,
  and `"Open AI"` all resolve to canonical `"OpenAI"`.

## Phase V & VI — Anti-Bot Strategy + Architecture

See `architecture.pdf` (2 pages): scale strategy, 413/429 handling,
distributed freshness tracking, storage strategy (Postgres + Neo4j +
vector DB + Redis), and the anti-bot approach (aiohttp-first, Playwright
fallback tier with residential proxies + stealth patches for
Cloudflare/Datadome-protected sources).

## Project structure
