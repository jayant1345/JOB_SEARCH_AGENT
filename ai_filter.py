"""
ai_filter.py — Use Claude to score each job's relevance for JK Data Lab.
Sends a batch of jobs and returns scored + filtered list.
"""

import json
import logging
import re
import requests

import config

logger = logging.getLogger("JobAgent")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are a job relevance scorer for JK Data Lab, an AI & Data Science consultancy.

JK Data Lab services:
- Multi-Agent AI Systems (LangChain, OpenAI, Python)
- RAG & Document Intelligence (FAISS, Streamlit)
- NLP & Text Analytics (BERT, HuggingFace, scikit-learn)
- BI & Dashboards (Streamlit, Plotly, SQL, FastAPI)
- Predictive Analytics & ML
- Python Automation & ETL

Rate each job 1-10 for relevance. Score 7+ means "apply".
IMPORTANT: Score any non-technical job (sales, bidding, HR, admin, data entry) as 1-2.
Return ONLY valid JSON, no markdown, no explanation.
Format: [{"id":"...","score":8,"reason":"short reason","apply":true}, ...]"""

_TECH_KEYWORDS = [
    "python", "langchain", "machine learning", "deep learning",
    "nlp", "natural language", "rag", "openai", "huggingface",
    "scikit", "tensorflow", "pytorch", "fastapi", "streamlit",
    "data science", "data scientist", "data engineer", "data analyst",
    "llm", "chatbot", "etl", "automation", "generative ai",
    "large language model", "computer vision", "ai engineer",
    "ml engineer", "artificial intelligence",
]

# Words that disqualify a job regardless of keyword hits
_DISQUALIFY = [
    "online bidder", "bid on projects", "sales executive", "telecaller",
    "bpo", "data entry", "apply for this position", "upload cv/resume",
    "cover letter *", "full name *email *",
]


def _is_irrelevant(job: dict) -> bool:
    """Quick pre-filter to drop obviously non-technical jobs before AI scoring."""
    text = (job["title"] + " " + job["description"]).lower()
    return any(phrase in text for phrase in _DISQUALIFY)


def _fallback_score(job: dict) -> dict:
    """
    Word-boundary keyword scoring used when Claude API is unavailable.
    Short terms (≤3 chars) use \\b word boundary to avoid false matches
    like 'ai' inside 'training' or 'data' inside 'storage and handling of your data'.
    """
    title_desc = (job["title"] + " " + job["description"]).lower()

    hits = 0
    for kw in _TECH_KEYWORDS:
        if len(kw) <= 3:
            if re.search(r"\b" + re.escape(kw) + r"\b", title_desc):
                hits += 1
        else:
            if kw in title_desc:
                hits += 1

    score = min(10, hits * 2)
    apply = hits >= 2  # require at least 2 specific tech terms (was 3 raw hits)
    job["ai_score"]  = score
    job["ai_reason"] = f"Keyword hits: {hits} ({', '.join(kw for kw in _TECH_KEYWORDS if kw in title_desc)[:60]})"
    job["ai_apply"]  = apply
    return job


_BATCH_SIZE = 5  # send 5 jobs per Claude call to avoid response truncation


def _score_batch(batch: list) -> None:
    """Send one batch of ≤5 jobs to Claude and mutate them in-place."""
    slim = [
        {
            "id": j["id"],
            "title": j["title"],
            "desc": j["description"][:200],
            "budget": j["budget"],
            "platform": j["platform"],
        }
        for j in batch
    ]
    prompt = f"Score these {len(slim)} jobs:\n{json.dumps(slim, ensure_ascii=False)}"

    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-haiku-4-5-20251001",  # faster + cheaper for scoring
                "max_tokens": 800,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("` \n")

        scores    = json.loads(raw)
        score_map = {s["id"]: s for s in scores}

        for job in batch:
            s = score_map.get(job["id"])
            if s:
                job["ai_score"]  = s.get("score", 5)
                job["ai_reason"] = s.get("reason", "")
                job["ai_apply"]  = s.get("apply", job.get("ai_score", 0) >= 7)
            else:
                _fallback_score(job)

        logger.info(f"Claude scored batch of {len(batch)} jobs")

    except Exception as e:
        logger.warning(f"Claude batch failed ({type(e).__name__}: {e}) — keyword fallback for this batch")
        for job in batch:
            _fallback_score(job)


def ai_score_jobs(jobs: list) -> list:
    """
    Score jobs with Claude (batched). Falls back to keyword scoring if API unavailable.
    """
    if not jobs:
        return []

    # Drop obviously irrelevant jobs before calling the API
    filtered = [j for j in jobs if not _is_irrelevant(j)]
    skipped  = len(jobs) - len(filtered)
    if skipped:
        logger.info(f"Pre-filter removed {skipped} non-technical jobs")
        for j in jobs:
            if _is_irrelevant(j):
                j["ai_score"] = 1
                j["ai_reason"] = "Non-technical job — skipped"
                j["ai_apply"] = False

    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — using keyword fallback for all jobs")
        for job in filtered:
            _fallback_score(job)
        return jobs

    logger.info(f"Sending {len(filtered)} jobs to Claude in batches of {_BATCH_SIZE}...")
    for i in range(0, len(filtered), _BATCH_SIZE):
        _score_batch(filtered[i : i + _BATCH_SIZE])

    return jobs
