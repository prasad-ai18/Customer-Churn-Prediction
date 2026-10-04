import sys
from pathlib import Path

# Ensure project root is in sys.path for Vercel Serverless Function imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.app import app

# Vercel Serverless Function entry point
# Exposes ASGI app instance
