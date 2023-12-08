"""Constants for ML"""

from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED, KEYS_TRAIN_NUM,
                                           KEY_TRAIN_TIME_DIFF, KEYS_TRAIN_NUM_TGT)

KEYS_EXTRACT_LIN = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED] + KEYS_TRAIN_NUM  # define cols to keep
KEYS_FEAT_SRC_LIN = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
KEYS_FEAT_TGT_LIN = KEYS_TRAIN_NUM_TGT + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
KEYS_EXTRACT_SEQ2SEQ = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED] + KEYS_TRAIN_NUM  # define cols to keep


