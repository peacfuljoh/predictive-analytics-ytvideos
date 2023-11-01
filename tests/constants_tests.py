
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
        users=dict(
            username=['username1', 'username2']
        ),
        video_meta=dict(
            video_id=[],
            username=[],
            title=[],
            upload_date=[],
            timestamp_first_seen=[],
            duration=[],
            keywords=[],
            description=[],
            thumbnail_url=[],
            tags=[]
        ),
        video_stats=dict(
            video_id=[],
            timestamp_accessed=[],
            like_count=[],
            view_count=[],
            subscriber_count=[],
            comment_count=[],
            comment=[]
        )
    )