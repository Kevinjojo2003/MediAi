"""Runtime configuration. Secrets are read from environment variables (or a local .env file)."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional; plain environment variables still work
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
