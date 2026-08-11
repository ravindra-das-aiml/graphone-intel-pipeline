# GraphOne / FrontierAtlas — Intelligence Graph Pipeline (Demo Task)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your real API keys
```

## Phase I — Research Papers Scraper

```bash
cd src/scrapers
python arxiv_scraper.py --query "cat:cs.AI" --max-results 1000 --output ../../data/research_papers.json
```

- Pulls papers from the Arxiv API (no auth needed, no Cloudflare — good
  first target since it's reliable).
- For each paper, looks up an associated GitHub repo via the Papers
  with Code API, then fetches its **current** star count from the
  GitHub REST API.
- Concurrency is controlled by a semaphore (`CONCURRENCY` in the
  script) so it doesn't trip 429s. Every request has exponential
  backoff + jitter built in.
- Output matches the `RESEARCH_PAPER` schema in the task brief
  exactly, written as a JSON array to `data/research_papers.json`.
- To scale toward 500k+: raise `CONCURRENCY`, run multiple category
  queries in parallel processes/machines, and swap the single JSON
  file sink for a queue (e.g. SQS/Kafka) writing into a database. No
  logic changes needed — just infrastructure.

## Project structure

```
src/
  scrapers/       # Phase I & II — data acquisition
  utils/          # shared helpers (retry, dedupe, etc.)
data/             # scraper output (JSON), not committed if large
architecture.pdf  # Phase VI design doc (added separately)
```

## Status

- [x] Phase I — Research papers scraper (Arxiv + GitHub stars)
- [ ] Phase I — Startups & products scraper
- [ ] Phase II — News + jobs freshness pipeline
- [ ] Phase III — LLM fallback extraction engine
- [ ] Phase IV — Entity resolution
- [ ] Phase V — Anti-bot documentation
- [ ] Phase VI — Architecture doc
