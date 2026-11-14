"""Runtime configuration. Secrets are read from environment variables (or a local .env file)."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass

def _get(name):
    """Environment variable first, then Streamlit secrets (used on Streamlit Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, "")
    except Exception:
        return ""


GEMINI_API_KEY = _get("GEMINI_API_KEY")
HUGGINGFACE_TOKEN = _get("HUGGINGFACE_TOKEN")
