"""Constants for ML"""

from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED, KEYS_TRAIN_NUM,
                                           KEY_TRAIN_TIME_DIFF, KEYS_TRAIN_NUM_TGT)

# keys
KEYS_EXTRACT_LIN = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED] + KEYS_TRAIN_NUM  # define cols to keep
KEYS_FEAT_SRC_LIN = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
KEYS_FEAT_TGT_LIN = KEYS_TRAIN_NUM_TGT + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
KEYS_EXTRACT_SEQ2SEQ = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED] + KEYS_TRAIN_NUM  # define cols to keep

# dataset
COL_SEQ_LEN_ORIG = 'seq_len_orig'
COL_SEQ_INFO_GROUP_ID = 'group_id'
COL_SEQ_LEN_GROUP = 'seq_len_group'

SEQ_LEN_GROUP_WIDTH = 5 # width of divisions along seq len dimension
TRAIN_BATCH_SIZE = 5
