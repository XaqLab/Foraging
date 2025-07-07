import json
import logging.config
import os
from pathlib import Path

from dotenv import load_dotenv

# Load config
current_dir = Path(__file__).resolve().parent
load_dotenv(current_dir / ".env")
with open(current_dir / os.getenv("LOGCONFIG_PATH"), "rb") as fp:
    logging.config.dictConfig(json.load(fp))
