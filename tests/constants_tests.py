

from src.crawler.crawler.constants import (COL_USERNAME, COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED, COL_TITLE, COL_KEYWORDS,
                                           COL_TAGS, COL_COMMENT, COL_DESCRIPTION, COL_DURATION, COL_COMMENT_COUNT,
                                           COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT, COL_LIKE_COUNT, COL_UPLOAD_DATE,
                                           COL_TIMESTAMP_FIRST_SEEN, COL_THUMBNAIL_URL)


DB_VIDEOS_DATABASE = "test_videos4645636"
DB_VIDEOS_NOSQL_DATABASE = "test_videos825463"
DB_FEATURES_NOSQL_DATABASE = "test_features2342422"
DB_MODELS_NOSQL_DATABASE = "test_models5857474"

ETL_CONFIG_NAME_PREFEATURES_TEST = 'test347583756'

try: # local
    from src.crawler.crawler.config import DB_INFO

    db_info = {
        "db_mysql_config": DB_INFO['DB_CONFIG'],
        "db_mongo_config": DB_INFO['DB_MONGO_CONFIG'],
        "db_info": {
            "DB_VIDEOS_DATABASE": DB_VIDEOS_DATABASE,
            "DB_VIDEOS_TABLES": {
                "users": "users",
                "meta": "video_meta",
                "stats": "video_stats"
            },
            "DB_VIDEOS_NOSQL_DATABASE": DB_VIDEOS_NOSQL_DATABASE,
            "DB_VIDEOS_NOSQL_COLLECTIONS": {
                "thumbnails": "thumbnails"
            },
            "DB_FEATURES_NOSQL_DATABASE": DB_FEATURES_NOSQL_DATABASE,
            "DB_FEATURES_NOSQL_COLLECTIONS": {
                "prefeatures": "prefeatures",
                "features": "features",
                "vocabulary": "vocabulary",
                "etl_config_prefeatures": "etl_config_prefeatures",
                "etl_config_features": "etl_config_features",
                "etl_config_vocabulary": "etl_config_vocabulary"
            },
            "DB_MODELS_NOSQL_DATABASE": DB_MODELS_NOSQL_DATABASE,
            "DB_MODELS_NOSQL_COLLECTIONS": {
                "models": "models",
                "meta": "metadata"
            }
        }
    }
except: # CI/CD
    pass



SCHEMA_SQL_ORIG_FNAME = 'src/crawler/crawler/dbs/ytvideos.sql'
SCHEMA_SQL_TEST_FNAME = 'tests/schema.sql'

DATA_SQL_TEST = dict(
        users={
            COL_USERNAME: ['username1', 'username2']
        },
        video_meta={
            COL_VIDEO_ID: [], # TODO: implement this
            COL_USERNAME: [],
            COL_TITLE: [],
            COL_UPLOAD_DATE: [],
            COL_TIMESTAMP_FIRST_SEEN: [],
            COL_DURATION: [],
            COL_KEYWORDS: [],
            COL_DESCRIPTION: [],
            COL_THUMBNAIL_URL: [],
            COL_TAGS: []
        },
        video_stats={
            COL_VIDEO_ID: [],
            COL_TIMESTAMP_ACCESSED: [],
            COL_LIKE_COUNT: [],
            COL_VIEW_COUNT: [],
            COL_SUBSCRIBER_COUNT: [],
            COL_COMMENT_COUNT: [],
            COL_COMMENT: []
        }
    )