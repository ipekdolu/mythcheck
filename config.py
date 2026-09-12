"""Shared environment/config loading for all pipeline stages."""
import os
import sys

import truststore
from dotenv import load_dotenv

# Mythology entity names routinely include diacritics (Odin's "Vali", Norse
# "AEsir", Egyptian "Ma'at", etc.); Windows' default console codepage (cp1252)
# can't encode many of them, which crashes plain print() calls.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Use the OS certificate store (Windows/macOS/Linux) instead of the bundled
# certifi list. Needed because some local networks/security software present
# certificate chains (e.g. via HTTPS inspection) that are trusted by the OS
# but not by certifi's Mozilla-derived bundle.
truststore.inject_into_ssl()

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

LANGFUSE_PUBLIC_KEY = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SECRET_KEY = os.environ["LANGFUSE_SECRET_KEY"]
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EXTRACTION_CACHE_DIR = os.environ.get("EXTRACTION_CACHE_DIR", "./.cache/extraction")

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
