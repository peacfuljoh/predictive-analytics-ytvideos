"""Config load and partitioning."""

from typing import Dict
import json
import os



try: # local
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
    DB_MYSQL_CONFIG: Dict[str, str] = config['DB_MYSQL_CONFIG']
    DB_MONGO_CONFIG: Dict[str, str] = config['DB_MONGO_CONFIG']
    AUTOCRAWL_CONFIG: dict = config['AUTOCRAWL_CONFIG']
    API_CONFIG: dict = config['API_CONFIG']

    # database info
    DB_INFO: dict = config['DB_INFO']
except: # CI/CD
    DB_MYSQL_CONFIG = dict(
        host="localhost",
        user=os.environ['MYSQL_USERNAME'],
        password=os.environ['MYSQL_PASSWORD']
    )
    DB_MONGO_CONFIG = dict(
        host=os.environ['MONGODB_HOST'],
        port=int(os.environ['MONGODB_PORT'])
    )
    REPO_ROOT = '/home/runner/work/predictive-analytics-ytvideos/predictive-analytics-ytvideos'
