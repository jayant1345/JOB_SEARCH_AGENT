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

# Technical keywords for fallback scoring — specific enough to avoid false matches
_TECH_KEYWORDS = [
    "python", "langchain", "machine learning", "deep learning",
    "nlp", "natural language", "rag", "openai", "huggingface",
    "scikit", "tensorflow", "pytorch", "fastapi", "streamlit",
    "data science", "llm", "chatbot", "etl", "automation",
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


def ai_score_jobs(jobs: list) -> list:
    """
    Score jobs with Claude. Returns enriched list with score/apply fields.
    Falls back to keyword scoring if API is unavailable or key not set.
    """
    if not jobs:
        return []

    # Drop obviously irrelevant jobs before even calling the API
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
        logger.warning("ANTHROPIC_API_KEY not set in config.py — using keyword fallback")
        for job in filtered:
            _fallback_score(job)
        return jobs

    # Build compact payload to save tokens
    slim = [
        {
            "id": j["id"],
            "title": j["title"],
            "desc": j["description"][:200],
            "budget": j["budget"],
            "platform": j["platform"],
        }
        for j in filtered
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
                "model": "claude-sonnet-4-6",
                "max_tokens": 1500,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores    = json.loads(raw)
        score_map = {s["id"]: s for s in scores}

        for job in filtered:
            s = score_map.get(job["id"], {})
            job["ai_score"]  = s.get("score", 5)
            job["ai_reason"] = s.get("reason", "")
            job["ai_apply"]  = s.get("apply", job["ai_score"] >= 7)

        logger.info(f"Claude scored {len(filtered)} jobs successfully")

    except Exception as e:
        logger.warning(f"AI scoring failed ({e}), using keyword fallback")
        for job in filtered:
            _fallback_score(job)

    return jobs
