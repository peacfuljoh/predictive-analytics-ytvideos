"""Config load and partitioning."""

from typing import Dict
import json
import os


# points to a JSON file outside of the repo that contains all secrets e.g. (DB credentials, paths, etc.)
GLOBAL_CONFIG_PATH = '/home/nuc/crawler_config/config.json'

# load the config
with open(GLOBAL_CONFIG_PATH, 'r') as f:
    config = json.load(f)

# root paths
DATA_ROOT: str = config['DATA_ROOT']
REPO_ROOT: str = config['REPO_ROOT']
CRAWLER_ROOT: str = os.path.join(REPO_ROOT, 'src', 'crawler', 'crawler')

# config info
DB_CONFIG: Dict[str, str] = config['DB_CONFIG']
DB_MONGO_CONFIG: Dict[str, str] = config['DB_MONGO_CONFIG']
AUTOCRAWL_CONFIG: dict = config['AUTOCRAWL_CONFIG']
API_CONFIG: dict = config['API_CONFIG']

# database info
DB_INFO: dict = config['DB_INFO']
