"""Miscellaneous constants for crawler"""

# settings during request to video stats table
MOST_RECENT_VID_LIMIT: int = 30
MAX_LEN_DESCRIPTION = 500
MAX_NUM_TAGS = 20
MAX_LEN_TAG = 25
MAX_NUM_KEYWORDS = 25
MAX_LEN_KEYWORD = 40

VIDEO_STATS_CAPTURE_WINDOW_DAYS = 5 # number of days into the past to consider current videos

# column name macros in raw data
COL_VIDEO_URL = 'video_url'
COL_UPLOAD_DATE = 'upload_date'
COL_TIMESTAMP_FIRST_SEEN = 'timestamp_first_seen'
COL_DURATION = 'duration'
COL_VIDEO_ID = 'video_id'
COL_USERNAME = 'username'
COL_LIKE_COUNT = 'like_count'
COL_COMMENT_COUNT = 'comment_count'
COL_SUBSCRIBER_COUNT = 'subscriber_count'
COL_VIEW_COUNT = 'view_count'
COL_TIMESTAMP_ACCESSED = 'timestamp_accessed'
COL_COMMENT = 'comment'
COL_TITLE = 'title'
COL_KEYWORDS = 'keywords'
COL_DESCRIPTION = 'description'
COL_TAGS = 'tags'
COL_THUMBNAIL_URL = 'thumbnail_url'

# Column names for 'stats' table
STATS_NUMERICAL_COLS = [COL_LIKE_COUNT, COL_COMMENT_COUNT, COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT]
STATS_DATETIME_COLS = [COL_TIMESTAMP_ACCESSED]
STATS_TEXT_COLS = [COL_VIDEO_ID, COL_COMMENT]
STATS_ALL_COLS = STATS_NUMERICAL_COLS + STATS_DATETIME_COLS + STATS_TEXT_COLS

# column names for 'meta' table
META_NUMERICAL_COLS = [COL_DURATION]
META_DATETIME_COLS = [COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN]
META_TEXT_COLS = [COL_VIDEO_ID, COL_USERNAME, COL_TITLE, COL_KEYWORDS, COL_DESCRIPTION, COL_TAGS]
META_ALL_COLS_NO_URL = META_NUMERICAL_COLS + META_DATETIME_COLS + META_TEXT_COLS
META_URL_COLS = [COL_THUMBNAIL_URL]
META_ALL_COLS = META_ALL_COLS_NO_URL + META_URL_COLS

# column names for 'prefeatures' collection (apart from raw data cols)
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

# column names for 'model' collection
MODEL_MODEL_OBJ = 'model'
MODEL_META_ID = 'meta_id'
MODEL_SPLIT_NAME = 'name'

# timestamps
TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f'
DATE_FMT = '%Y-%m-%d'
TIMESTAMP_CONVERSION_FMTS = {
    COL_TIMESTAMP_ACCESSED: TIMESTAMP_FMT,
    COL_TIMESTAMP_FIRST_SEEN: TIMESTAMP_FMT,
    COL_UPLOAD_DATE: DATE_FMT
}

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
KEYS_TRAIN_ID = [COL_USERNAME, COL_VIDEO_ID]
KEYS_TRAIN_NUM = [COL_COMMENT_COUNT, COL_LIKE_COUNT, COL_VIEW_COUNT, COL_SUBSCRIBER_COUNT]
KEYS_TRAIN_NUM_TGT = [key for key in KEYS_TRAIN_NUM if key != COL_SUBSCRIBER_COUNT]
KEY_TRAIN_TIME_DIFF = 'time_after_upload' # seconds

# dict key names for encoded model
MODEL_DICT_PREPROCESSOR = 'preprocessor'
MODEL_DICT_DATA_BOW = 'data_bow'
MODEL_DICT_MODEL = 'model'
MODEL_DICT_CONFIG = 'config'

# col names for DataFrames in LR model
KEYS_FOR_FIT_NONBOW_SRC = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]
KEYS_FOR_FIT_NONBOW_SRC = [key + '_src' for key in KEYS_FOR_FIT_NONBOW_SRC] + [KEY_TRAIN_TIME_DIFF + '_tgt']
KEYS_FOR_FIT_NONBOW_TGT = KEYS_TRAIN_NUM_TGT
KEYS_FOR_FIT_NONBOW_TGT = [key + '_tgt' for key in KEYS_FOR_FIT_NONBOW_TGT]
KEYS_FOR_PRED_NONBOW_ID = KEYS_TRAIN_ID + [KEY_TRAIN_TIME_DIFF + suffix for suffix in ['_src', '_tgt']]
KEYS_FOR_PRED_NONBOW_TGT = [key + '_pred' for key in KEYS_TRAIN_NUM_TGT]

# ETL config key info
ETL_CONFIG_VALID_KEYS_PREFEATURES = dict(
    extract=['filters', 'limit'],
    transform=['include_additional_keys'],
    load=[],
    preconfig=[]
)
ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES = dict(
    extract=['filters', 'limit'],
    transform=[],
    load=[],
    preconfig=[]
)
ETL_CONFIG_VALID_KEYS_VOCAB = dict(
    extract=['filters', PREFEATURES_ETL_CONFIG_COL],
    transform=[],
    load=[],
    preconfig=[PREFEATURES_ETL_CONFIG_COL]
)
ETL_CONFIG_EXCLUDE_KEYS_VOCAB = dict(
    extract=[],
    transform=[],
    load=[],
    preconfig=[]
)
# VOCAB_ETL_
ETL_CONFIG_VALID_KEYS_FEATURES = dict(
    extract=['filters', PREFEATURES_ETL_CONFIG_COL],
    transform=[],
    load=[],
    preconfig=[PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL]
)
ETL_CONFIG_EXCLUDE_KEYS_FEATURES = dict(
    extract=[],
    transform=[],
    load=[],
    preconfig=[]
)

# web sockets
WS_STREAM_TERM_MSG = 'DONE'

WS_MAX_RECORDS_SEND = 100
