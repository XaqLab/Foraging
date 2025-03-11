from dotenv import load_dotenv
import logging.config
import os
import json
from pathlib import Path

# Load config
current_dir = Path(__file__).resolve().parent
load_dotenv(current_dir / '.env')
with open(current_dir / os.getenv('LOGCONFIG_PATH'), 'rb') as fp:
    logging.config.dictConfig(json.load(fp))