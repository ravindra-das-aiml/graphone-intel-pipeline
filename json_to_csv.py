"""
Converts all Phase I-IV JSON outputs into flat CSV files, one per required
Google Sheet tab (Startups, Products, Research Papers, Jobs, News,
Entity Mapping Log), ready for File > Import in Google Sheets.

Run:
    python json_to_csv.py
"""

import csv
import json
import os

DATA_DIR = "data"
OUT_DIR = "data/csv_for_sheets"


def load(path):
    full = os.path.join(DATA_DIR, path)
    if not os.path.exists(full):
        print(f"WARNING: {full} not found, skipping.")
        return []
    with open(full, encoding="utf-8") as f:
        return json.load(f)


def write_csv(filename, rows, fieldnames):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows -> {path}")


def flatten_startups():
    data = load("startups.json")
    rows = []
    for s in data:
        c = s["content"]
        rows.append({
            "schemaVersion": s["schemaVersion"], "recordType": s["recordType"],
            "source_name": s["source"]["name"], "source_url": s["source"]["url"],
            "entityName": c["entityName"],
            "employeeCount": c["data"]["employeeCount"] if c.get("data") else "",
            "collectedAt": s["collectedAt"],
        })
    write_csv("Startups.csv", rows,
               ["schemaVersion", "recordType", "source_name", "source_url",
                "entityName", "employeeCount", "collectedAt"])


def flatten_products():
    data = load("products.json")
    rows = []
    for p in data:
        c = p["content"]
        rows.append({
            "schemaVersion": p["schemaVersion"], "recordType": p["recordType"],
            "source_name": p["source"]["name"], "source_url": p["source"]["url"],
            "startupName": c["startupName"], "pricingModel": c["pricingModel"] or "",
            "collectedAt": p["collectedAt"],
        })
    write_csv("Products.csv", rows,
               ["schemaVersion", "recordType", "source_name", "source_url",
                "startupName", "pricingModel", "collectedAt"])


def flatten_papers():
    data = load("research_papers.json")
    rows = []
    for r in data:
        c = r["content"]
        rows.append({
            "schemaVersion": r["schemaVersion"], "recordType": r["recordType"],
            "source_name": r["source"]["name"], "source_url": r["source"]["url"],
            "title": c["title"], "authors": "; ".join(c["authors"]),
            "paper_url": c["paper_url"], "github_url": c["github_url"] or "",
            "github_stars": c["github_stars"] if c["github_stars"] is not None else "",
            "published_date": c["published_date"], "collectedAt": r["collectedAt"],
        })
    write_csv("Research_Papers.csv", rows,
               ["schemaVersion", "recordType", "source_name", "source_url", "title",
                "authors", "paper_url", "github_url", "github_stars",
                "published_date", "collectedAt"])


def flatten_jobs():
    data = load("jobs.json")
    rows = []
    for j in data:
        c = j["content"]
        rows.append({
            "schemaVersion": j["schemaVersion"], "recordType": j["recordType"],
            "source_name": j["source"]["name"], "source_url": j["source"]["url"],
            "company": c["company"], "title": c.get("title", ""),
            "date": c["date"], "is_remote": c["is_remote"],
            "role_family": c["role_family"], "collectedAt": j["collectedAt"],
        })
    write_csv("Jobs.csv", rows,
               ["schemaVersion", "recordType", "source_name", "source_url",
                "company", "title", "date", "is_remote", "role_family", "collectedAt"])


def flatten_news():
    data = load("news_enriched.json") or load("news.json")
    rows = []
    for n in data:
        c = n["content"]
        extraction = c.get("llm_extraction") or {}
        rows.append({
            "schemaVersion": n["schemaVersion"], "recordType": n["recordType"],
            "source_name": n["source"]["name"], "source_url": n["source"]["url"],
            "title": c["title"],
            "full_text_preview": (c["full_text"][:300] + "...") if len(c["full_text"]) > 300 else c["full_text"],
            "published_date": c["published_date"],
            "companies_mentioned": "; ".join(extraction.get("companies_mentioned", [])),
            "topic": extraction.get("topic", ""), "sentiment": extraction.get("sentiment", ""),
            "collectedAt": n["collectedAt"],
        })
    write_csv("News.csv", rows,
               ["schemaVersion", "recordType", "source_name", "source_url", "title",
                "full_text_preview", "published_date", "companies_mentioned",
                "topic", "sentiment", "collectedAt"])


def flatten_entity_log():
    data = load("entity_mapping_log.json")
    rows = []
    for e in data:
        rows.append({
            "raw_name": e["raw_name"], "canonical_name": e["canonical_name"] or "(unmatched)",
            "method": e["method"], "confidence": e["confidence"],
            "source": e["source"], "resolvedAt": e["resolvedAt"],
        })
    write_csv("Entity_Mapping_Log.csv", rows,
               ["raw_name", "canonical_name", "method", "confidence", "source", "resolvedAt"])


if __name__ == "__main__":
    flatten_startups()
    flatten_products()
    flatten_papers()
    flatten_jobs()
    flatten_news()
    flatten_entity_log()
    print(f"\nAll CSVs are in {OUT_DIR}/ — import each one as a separate tab in Google Sheets.")