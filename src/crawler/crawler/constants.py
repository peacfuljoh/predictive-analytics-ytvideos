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
PREFEATURES_ETL_CONFIG_COL = 'etl_config_prefeatures'
PREFEATURES_TOKENS_COL = 'tokens'

# column names for 'vocabulary' collection
VOCAB_VOCABULARY_COL = 'vocabulary'
VOCAB_TIMESTAMP_COL = 'timestamp_vocabulary'
VOCAB_ETL_CONFIG_COL = 'etl_config_vocabulary'

# column names for 'features' collection
FEATURES_VECTOR_COL = 'vec'
FEATURES_TIMESTAMP_COL = 'timestamp_features'
FEATURES_ETL_CONFIG_COL = 'etl_config_features'

# other
VIDEO_STATS_CAPTURE_WINDOW_DAYS = 5 # number of days into the past to consider current videos
TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f'

# ML model
MIN_VID_SAMPS_FOR_DATASET = 10
NUM_INTVLS_PER_VIDEO = 100
VEC_EMBED_DIMS = 512
TRAIN_TEST_SPLIT_DFLT = 0.8

ML_MODEL_TYPE = 'model_type'
ML_MODEL_HYPERPARAMS = 'hyperparams'
ML_MODEL_TYPE_LIN_PROJ_RAND = 'lin_proj_random'
ML_MODEL_TYPE_GAM_TOPIC = 'gam_topic'
ML_HYPERPARAM_EMBED_DIM = 'embed_dim'
ML_HYPERPARAM_RLP_DENSITY = 'lin_proj_random_density'
ML_HYPERPARAM_SR_ALPHAS = 'simple_reg_alphas'
ML_HYPERPARAM_SR_CV_SPLIT = 'simple_reg_cv_split'
ML_HYPERPARAM_SR_CV_COUNT = 'simple_reg_cv_count'
SPLIT_TRAIN_BY_USERNAME = 'split_train_by_username'
ML_CONFIG_KEYS = [ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS]
ML_MODEL_TYPES = [ML_MODEL_TYPE_LIN_PROJ_RAND, ML_MODEL_TYPE_GAM_TOPIC]
TRAIN_TEST_SPLIT = 'tt_split'
KEYS_TRAIN_ID = ['username', 'video_id']
KEYS_TRAIN_NUM = ['comment_count', 'like_count', 'view_count', 'subscriber_count']
KEYS_TRAIN_NUM_TGT = [key for key in KEYS_TRAIN_NUM if key != 'subscriber_count']
KEY_TRAIN_TIME_DIFF = 'time_after_upload' # seconds

KEYS_FOR_FIT_NONBOW_SRC = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]
KEYS_FOR_FIT_NONBOW_SRC = [key + '_src' for key in KEYS_FOR_FIT_NONBOW_SRC] + [KEY_TRAIN_TIME_DIFF + '_tgt']
KEYS_FOR_FIT_NONBOW_TGT = KEYS_TRAIN_NUM_TGT
KEYS_FOR_FIT_NONBOW_TGT = [key + '_tgt' for key in KEYS_FOR_FIT_NONBOW_TGT]
KEYS_FOR_PRED_NONBOW_ID = KEYS_TRAIN_ID + [KEY_TRAIN_TIME_DIFF + suffix for suffix in ['_src', '_tgt']]
KEYS_FOR_PRED_NONBOW_TGT = [key + '_pred' for key in KEYS_TRAIN_NUM_TGT]
