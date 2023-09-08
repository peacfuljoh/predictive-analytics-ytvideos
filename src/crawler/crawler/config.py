
from typing import Dict
import json
import os


# points to a JSON file outside of the repo that contains all secrets e.g. (DB credentials, paths, etc.)
GLOBAL_CONFIG_PATH = '/home/nuc/crawler_config/config.json'

# load the
with open(GLOBAL_CONFIG_PATH, 'r') as f:
    config = json.load(f)

DB_CONFIG: Dict[str, str] = config['DB_CONFIG']
DATA_ROOT: str = config['DATA_ROOT']
REPO_ROOT: str = config['REPO_ROOT']
CRAWLER_ROOT: str = os.path.join(REPO_ROOT, 'src', 'crawler', 'crawler')
DB_INFO: dict = config['DB_INFO']
