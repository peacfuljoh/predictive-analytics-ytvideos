"""Miscellaneous constants for crawler"""

# settings during request to video stats table
MOST_RECENT_VID_LIMIT: int = 30
DB_KEY_UPLOAD_DATE: str = 'upload_date'
DB_KEY_TIMESTAMP_FIRST_SEEN: str = 'timestamp_first_seen'
VIDEO_URL_COL_NAME: str = 'video_url'
MAX_LEN_DESCRIPTION = 500
MAX_NUM_TAGS = 20
MAX_LEN_TAG = 25
MAX_NUM_KEYWORDS = 25
MAX_LEN_KEYWORD = 40

# Column names for 'stats' table
STATS_NUMERICAL_COLS = ['like_count', 'comment_count', 'subscriber_count', 'view_count']
STATS_DATETIME_COLS = ['timestamp_accessed']
STATS_TEXT_COLS = ['video_id', 'comment']
STATS_ALL_COLS = STATS_NUMERICAL_COLS + STATS_DATETIME_COLS + STATS_TEXT_COLS

# column names for 'meta' table
META_NUMERICAL_COLS = ['duration']
META_DATETIME_COLS = ['upload_date', 'timestamp_first_seen']
META_TEXT_COLS = ['video_id', 'username', 'title', 'keywords', 'description', 'tags']
META_ALL_COLS_NO_URL = META_NUMERICAL_COLS + META_DATETIME_COLS + META_TEXT_COLS
META_URL_COLS = ['thumbnail_url']
META_ALL_COLS = META_ALL_COLS_NO_URL + META_URL_COLS

# column names for 'prefeatures' collection
PREFEATURES_USERNAME_COL = 'username'
PREFEATURES_TIMESTAMP_COL = 'timestamp_accessed'
PREFEATURES_VIDEO_ID_COL = 'video_id'
PREFEATURES_ETL_CONFIG_COL = 'etl_config'
PREFEATURES_TOKENS_COL = 'tokens'

# column names for 'vocabulary' collection
VOCAB_VOCABULARY_COL = 'vocabulary'
VOCAB_TIMESTAMP_COL = 'timestamp'
VOCAB_ETL_CONFIG_COL = 'etl_config_vocabulary'

# column names for 'features' collection
FEATURES_VECTOR_COL = 'vec'
FEATURES_ETL_CONFIG_COL = 'etl_config_features'

# other
VIDEO_STATS_CAPTURE_WINDOW_DAYS = 3 # number of days into the past to consider current videos
TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f'
