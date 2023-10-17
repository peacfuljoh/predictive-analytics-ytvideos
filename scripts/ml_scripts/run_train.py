"""Script for testing train and test"""

import math

from src.ml.train_utils import (load_feature_records, train_test_split, prepare_feature_records,
                                train_regression_model_simple)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, VEC_EMBED_DIMS, TRAIN_TEST_SPLIT,
                                           ML_HYPERPARAM_SR_ALPHAS, ML_HYPERPARAM_SR_CV_SPLIT,
                                           ML_HYPERPARAM_SR_CV_COUNT)
from src.crawler.crawler.utils.misc_utils import print_df_full
from src.ml.ml_request import MLRequest



# specify which version of the feature store to use
config_load = {
    PREFEATURES_ETL_CONFIG_COL: 'test3',
    VOCAB_ETL_CONFIG_COL: 'test007',
    FEATURES_ETL_CONFIG_COL: 'test007',
    # FEATURES_TIMESTAMP_COL: '2023-10-05 17:13:16.668'
}

# specify ML model options
rlp_density = 0.01 # density of sparse projector matrix
train_test_split_fract = 0.8 # fraction of data for train
sr_alphas = [1e-6, 1e-5, 1e-4, 1e-3] # simple regression regularization coefficient
cv_split = 0.9 # cross-validation split ratio
cv_count = 10 # number of CV splits

config_ml = {
    ML_MODEL_TYPE: ML_MODEL_TYPE_LIN_PROJ_RAND,
    ML_MODEL_HYPERPARAMS: {
        ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS, # number of dimensions in dense vectors after projecting sparse bow vecs
        ML_HYPERPARAM_RLP_DENSITY: rlp_density,
        ML_HYPERPARAM_SR_ALPHAS: sr_alphas,
        ML_HYPERPARAM_SR_CV_SPLIT: cv_split,
        ML_HYPERPARAM_SR_CV_COUNT: cv_count
    },
    TRAIN_TEST_SPLIT: train_test_split_fract
}

ml_request = MLRequest(config_ml)

# see all configs
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# load feature records
df_gen, config_data = load_feature_records(config_load)

# preprocess feature records
data_all, model_embed = prepare_feature_records(df_gen, ml_request)

# split into train and test
train_test_split(data_all, ml_request) # in-place

# train model
model_reg = train_regression_model_simple(data_all, ml_request)

# evaluate on held-out test set

