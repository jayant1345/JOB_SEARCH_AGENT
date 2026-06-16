# ============================================================
# JK Data Lab — Job Search AI Agent Config
# Owner: Kinjal Jayantkumar Jayswal
# ============================================================

import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv()

# ── API Keys (loaded from .env) ───────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Notifications ─────────────────────────────────────────────
WHATSAPP_NUMBER    = "+919157938887"
TELEGRAM_BOT_TOKEN = ""   # Fill in after creating bot via @BotFather
TELEGRAM_CHAT_ID   = ""   # Fill in after /start with your bot

# ── Scan schedule ─────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 120   # every 2 hours

# ── Keyword filter (any one match is enough) ──────────────────
KEYWORDS = [
    "python", "ai", "artificial intelligence", "machine learning", "ml",
    "data science", "langchain", "rag", "llm", "openai", "nlp",
    "streamlit", "fastapi", "etl", "automation", "scraping",
    "chatbot", "agent", "data analyst", "deep learning", "huggingface",
]

# ── Client quality filters ────────────────────────────────────
MIN_CLIENT_RATING        = 4.0
MIN_PAID_PROJECTS        = 3
ZERO_RATING_BUDGET_LIMIT = 5000   # INR — skip unrated clients above this budget

# ── File paths ────────────────────────────────────────────────
STATE_FILE = "seen_jobs.json"
JOBS_FILE  = "jobs_found.json"
LOG_FILE   = "agent.log"
