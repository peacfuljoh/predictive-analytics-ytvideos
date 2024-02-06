"""Miscellaneous constants for crawler"""

import datetime

import pandas as pd



# settings during request to video stats table
MOST_RECENT_VID_LIMIT: int = 30
MAX_LEN_DESCRIPTION = 500
MAX_NUM_TAGS = 20
MAX_LEN_TAG = 25
MAX_NUM_KEYWORDS = 25
MAX_LEN_KEYWORD = 40

VIDEO_STATS_CAPTURE_WINDOW_DAYS = 7 # number of days into the past to consider current videos

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
TIMESTAMP_FMT_FPATH = '%Y-%m-%d_%H-%M-%S-%f'
TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%f'
DATE_FMT = '%Y-%m-%d'
func_encode_str = lambda df_col: df_col.astype(str)
TIMESTAMP_CONVERSION_FMTS_ENCODE = {
    COL_TIMESTAMP_ACCESSED: {'func': func_encode_str},
    COL_TIMESTAMP_FIRST_SEEN: {'func': func_encode_str},
    COL_UPLOAD_DATE: {'func': func_encode_str},
    VOCAB_TIMESTAMP_COL: {'func': func_encode_str},
    FEATURES_TIMESTAMP_COL: {'func': func_encode_str}
}
func_decode_date = lambda df_col: df_col.map(lambda s: datetime.datetime.strptime(s, DATE_FMT).date())
# func_decode_timestamp = lambda df_col: pd.to_datetime(df_col, format=TIMESTAMP_FMT)
def func_decode_timestamp_one(s):
    fmts = (TIMESTAMP_FMT, DATE_FMT, '%Y-%m-%d %H:%M:%S')
    for fmt in fmts:
        try:
            return datetime.datetime.strptime(s, fmt)
        except:
            pass
    raise Exception(f"No time format found that matches {s} among formats {fmts}")
func_decode_timestamp = lambda df_col: df_col.map(func_decode_timestamp_one) # TODO: vectorize by format instead of converting one-by-one
TIMESTAMP_CONVERSION_FMTS_DECODE = {
    COL_TIMESTAMP_ACCESSED: {'func': func_decode_timestamp},
    COL_TIMESTAMP_FIRST_SEEN: {'func': func_decode_timestamp},
    COL_UPLOAD_DATE: {'func': func_decode_date},
    VOCAB_TIMESTAMP_COL: {'func': func_decode_timestamp},
    FEATURES_TIMESTAMP_COL: {'func': func_decode_timestamp}
}

# ML model
MIN_VID_SAMPS_FOR_DATASET = 10
MIN_VID_SAMPS_FOR_DATASET_SEQ2SEQ = 24
NUM_INTVLS_PER_VIDEO = 100
VEC_EMBED_DIMS = 512
TRAIN_TEST_SPLIT_DFLT = 0.8
TRAIN_SEQ_PERIOD = 3600

ML_MODEL_TYPE = 'model_type'
ML_MODEL_HYPERPARAMS = 'hyperparams'
ML_MODEL_TYPE_LIN_PROJ_RAND = 'lin_proj_random' # linear regression model with random projection of bag-of-words
ML_MODEL_TYPE_SEQ2SEQ = 'seq2seq'
ML_HYPERPARAM_EMBED_DIM = 'embed_dim'
ML_HYPERPARAM_RLP_DENSITY = 'lin_proj_random_density'
ML_HYPERPARAM_SR_ALPHAS = 'simple_reg_alphas'
ML_HYPERPARAM_SR_CV_SPLIT = 'simple_reg_cv_split'
ML_HYPERPARAM_SR_CV_COUNT = 'simple_reg_cv_count'
SPLIT_TRAIN_BY_USERNAME = 'split_train_by_username'
ML_CONFIG_KEYS = [ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS]
ML_MODEL_TYPES = [ML_MODEL_TYPE_LIN_PROJ_RAND, ML_MODEL_TYPE_SEQ2SEQ]
TRAIN_TEST_SPLIT = 'tt_split'
KEYS_TRAIN_ID = [COL_USERNAME, COL_VIDEO_ID]
KEYS_TRAIN_NUM = [COL_COMMENT_COUNT, COL_LIKE_COUNT, COL_VIEW_COUNT, COL_SUBSCRIBER_COUNT]

NON_TGT_KEYS = [COL_SUBSCRIBER_COUNT]
KEYS_TRAIN_NUM_TGT = [key for key in KEYS_TRAIN_NUM if key not in NON_TGT_KEYS]
KEYS_TRAIN_NUM_STATIC_IDXS = [i for i, key in enumerate(KEYS_TRAIN_NUM) if key in NON_TGT_KEYS]
KEYS_TRAIN_NUM_TGT_IDXS = [i for i, key in enumerate(KEYS_TRAIN_NUM) if key not in NON_TGT_KEYS]
TRAIN_NUM_TGT_KEYS = [key for i, key in enumerate(KEYS_TRAIN_NUM) if i in KEYS_TRAIN_NUM_TGT_IDXS]
KEY_TRAIN_TIME_DIFF = 'time_after_upload' # seconds
VEC_EMBED_DIMS_NN = VEC_EMBED_DIMS + len(KEYS_TRAIN_NUM_STATIC_IDXS)

MODEL_ID = 'model_id'
COL_SEQ_SPLIT_FOR_PRED = 'seq_split_for_pred'
SEQ_SPLIT_INCLUDE = 'include'
SEQ_SPLIT_EXCLUDE = 'exclude'

# dict key names for encoded model
MODEL_DICT_PREPROCESSOR = 'preprocessor'
MODEL_DICT_DATA_BOW = 'data_bow'
MODEL_DICT_MODEL = 'model'
MODEL_DICT_CONFIG = 'config'

# col names for DataFrames in LR model
KEYS_FOR_FIT_NONBOW_SRC = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF] # excludes FEATURES_VECTOR_COL
KEYS_FOR_FIT_NONBOW_SRC = [key + '_src' for key in KEYS_FOR_FIT_NONBOW_SRC] + [KEY_TRAIN_TIME_DIFF + '_tgt']
KEYS_FOR_FIT_NONBOW_TGT = [key + '_tgt' for key in KEYS_TRAIN_NUM_TGT]
KEYS_FOR_PRED_NONBOW_ID = KEYS_TRAIN_ID + [KEY_TRAIN_TIME_DIFF + suffix for suffix in ['_src', '_tgt']]
KEYS_FOR_PRED_NONBOW_TGT = [key + '_pred' for key in KEYS_TRAIN_NUM_TGT]
assert [val[:-4] for val in KEYS_FOR_FIT_NONBOW_TGT] == [val[:-5] for val in KEYS_FOR_PRED_NONBOW_TGT]

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



