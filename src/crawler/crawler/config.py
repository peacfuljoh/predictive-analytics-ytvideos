"""Config load and partitioning."""

from typing import Dict
import json
import os



# test database and config names
DB_VIDEOS_DATABASE_TEST = "test_videos4645636"
DB_VIDEOS_NOSQL_DATABASE_TEST = "test_videos825463"
DB_FEATURES_NOSQL_DATABASE_TEST = "test_features2342422"
DB_MODELS_NOSQL_DATABASE_TEST = "test_models5857474"

ETL_CONFIG_NAME_PREFEATURES_TEST = 'test347583756'


# load config info
GLOBAL_CONFIG_PATH = '/home/nuc/crawler_config/config.json'
is_local = os.path.exists(GLOBAL_CONFIG_PATH)

if is_local: # local
    # points to a JSON file outside of the repo that contains all secrets e.g. (DB credentials, paths, etc.)
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
else: # CI/CD
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

# determine where to get DB_INFO
if 'RUN_TESTS' not in os.environ or os.environ['RUN_TESTS'] == 'no':
    if is_local: # not testing, running locally
        DB_INFO: dict = config['DB_INFO']
    else:
        raise NotImplementedError('Can only run non-tests in a local env.')
elif os.environ['RUN_TESTS'] == 'yes':
    # setup db info and configs
    DB_INFO = {
        "db_mysql_config": DB_MYSQL_CONFIG,
        "db_mongo_config": DB_MONGO_CONFIG,
        "db_info": {
            "DB_VIDEOS_DATABASE": DB_VIDEOS_DATABASE_TEST,
            "DB_VIDEOS_TABLES": {
                "users": "users",
                "meta": "video_meta",
                "stats": "video_stats"
            },
            "DB_VIDEOS_NOSQL_DATABASE": DB_VIDEOS_NOSQL_DATABASE_TEST,
            "DB_VIDEOS_NOSQL_COLLECTIONS": {
                "thumbnails": "thumbnails"
            },
            "DB_FEATURES_NOSQL_DATABASE": DB_FEATURES_NOSQL_DATABASE_TEST,
            "DB_FEATURES_NOSQL_COLLECTIONS": {
                "prefeatures": "prefeatures",
                "features": "features",
                "vocabulary": "vocabulary",
                "etl_config_prefeatures": "etl_config_prefeatures",
                "etl_config_features": "etl_config_features",
                "etl_config_vocabulary": "etl_config_vocabulary"
            },
            "DB_MODELS_NOSQL_DATABASE": DB_MODELS_NOSQL_DATABASE_TEST,
            "DB_MODELS_NOSQL_COLLECTIONS": {
                "models": "models",
                "meta": "metadata"
            }
        }
    }
else:
    raise Exception('Specify valid RUN_TESTS env var.')
