"""Build the retrieval corpus: every English Ask CFPB answer, as one JSONL snapshot.

    make corpus     # = uv run python -m complaint_routing.llm.corpus -> data/corpus/ask_cfpb.jsonl

Ask CFPB (consumerfinance.gov/ask-cfpb) is the CFPB's own consumer Q&A: one page per question,
grouped in categories that mirror the complaint products (debt collection, credit reports,
mortgages, ...). It is a US federal government work, public domain. The answer pages are listed
in the site's sitemap.xml with the suffix -en-<id>/; <id> becomes the document id `ask-<id>`.

The pages change over time ("last reviewed" dates), so the committed snapshot, not the live site,
is the corpus version; its md5 is logged on every LLM run.
"""

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

from complaint_routing.config import ROOT

SITEMAP = "https://www.consumerfinance.gov/sitemap.xml"
CORPUS_PATH = ROOT / "data" / "corpus" / "ask_cfpb.jsonl"
BOILERPLATE = re.compile(
    r"^(Don't see what you're looking for\?|Search for your question|Searches are limited to"
    r"|Browse related questions|Learn more about)"
)
ANSWER_URL = re.compile(r"https://www\.consumerfinance\.gov/ask-cfpb/[a-z0-9-]+-en-(\d+)/")


def fetch(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def answer_urls() -> list[tuple[str, str]]:
    xml = fetch(SITEMAP)
    found = {m.group(1): m.group(0) for m in ANSWER_URL.finditer(xml)}
    return sorted(((f"ask-{i}", u) for i, u in found.items()), key=lambda t: int(t[0][4:]))


def parse(html: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find("h1").get_text(" ", strip=True)
    crumbs = soup.select_one("nav.m-breadcrumbs")
    category = crumbs.get_text(" ", strip=True).strip("/ ").split(" / ")[-1] if crumbs else ""
    body = soup.select_one("div.answer-text")
    reviewed = soup.select_one("time.answer-edited-date")
    # Page furniture inside the answer: related-question links, "learn more", the search box.
    for junk in body.select(".related-questions, form, .m-global-search, .o-search-bar"):
        junk.decompose()
    lead = soup.select_one("div.lead-paragraph")  # the one-paragraph short answer
    # Paragraph-ish blocks joined by newlines, so the chunker sees the page structure.
    blocks = [lead.get_text(" ", strip=True)] if lead else []
    blocks += [
        e.get_text(" ", strip=True)
        for e in body.find_all(["p", "li", "h2", "h3", "h4", "td"])
        if e.get_text(strip=True)
    ]
    blocks = [b for b in blocks if not BOILERPLATE.search(b)]
    text = "\n".join(dict.fromkeys(blocks)) or body.get_text("\n", strip=True)
    return {
        "title": title,
        "category": category,
        "last_reviewed": reviewed.get("datetime", reviewed.get_text(strip=True))
        if reviewed
        else "",
        "text": re.sub(r"[ \t]+", " ", text),
    }


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def corpus_version(path: Path = CORPUS_PATH) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=CORPUS_PATH)
    p.add_argument("--delay", type=float, default=0.3, help="seconds between requests")
    args = p.parse_args()

    urls = answer_urls()
    print(f"{len(urls)} English Ask CFPB answers in the sitemap")
    docs = []
    for n, (doc_id, url) in enumerate(urls, 1):
        try:
            doc = {"doc_id": doc_id, "url": url} | parse(fetch(url))
        except Exception as e:  # a page removed between sitemap and fetch
            print(f"skip {doc_id}: {e}")
            continue
        if len(doc["text"]) >= 200:
            docs.append(doc)
        if n % 50 == 0:
            print(f"{n}/{len(urls)}")
        time.sleep(args.delay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"wrote {len(docs)} documents to {args.out} (md5 {corpus_version(args.out)})")


if __name__ == "__main__":
    main()
