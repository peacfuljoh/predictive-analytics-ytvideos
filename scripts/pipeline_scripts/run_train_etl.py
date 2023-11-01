"""Script for testing train and test"""

from src.ml.train_utils import (load_feature_records, train_test_split, prepare_feature_records,
                                train_regression_model_simple, save_reg_model)
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, VEC_EMBED_DIMS, TRAIN_TEST_SPLIT,
                                           ML_HYPERPARAM_SR_ALPHAS, ML_HYPERPARAM_SR_CV_SPLIT,
                                           ML_HYPERPARAM_SR_CV_COUNT, SPLIT_TRAIN_BY_USERNAME)
from ytpa_utils.misc_utils import print_df_full
from src.ml.ml_request import MLRequest
from src.crawler.crawler.config import DB_INFO, DB_MYSQL_CONFIG, DB_MONGO_CONFIG


db_ = {'db_info': DB_INFO, 'db_mysql_config': DB_MYSQL_CONFIG, 'db_mongo_config': DB_MONGO_CONFIG}


# specify which version of the feature store to use
config_load = {
    PREFEATURES_ETL_CONFIG_COL: 'test3',
    VOCAB_ETL_CONFIG_COL: 'test21715',
    FEATURES_ETL_CONFIG_COL: 'test21715',
    # FEATURES_TIMESTAMP_COL: '2023-10-05 17:13:16.668'
}

# specify ML model options
rlp_density = 0.01 # density of sparse projector matrix
train_test_split_fract = 0.8 # fraction of raw_data for train
sr_alphas = [1e-6, 1e-4, 1e-2] # simple regression regularization coefficient
cv_split = 0.9 # cross-validation split ratio
cv_count = 10 # number of CV splits
split_train_by_username = True

config_ml = {
    ML_MODEL_TYPE: ML_MODEL_TYPE_LIN_PROJ_RAND,
    ML_MODEL_HYPERPARAMS: {
        ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS, # number of dimensions in dense vectors after projecting sparse bow vecs
        ML_HYPERPARAM_RLP_DENSITY: rlp_density,
        ML_HYPERPARAM_SR_ALPHAS: sr_alphas,
        ML_HYPERPARAM_SR_CV_SPLIT: cv_split,
        ML_HYPERPARAM_SR_CV_COUNT: cv_count
    },
    TRAIN_TEST_SPLIT: train_test_split_fract,
    SPLIT_TRAIN_BY_USERNAME: split_train_by_username,
    'db': db_
}

ml_request = MLRequest(config_ml)

# see all configs
# from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# get data, train model
if 1:
    # load feature records
    df_gen, config_data = load_feature_records(config_load, ml_request)

    # preprocess feature records
    data_all, model_embed = prepare_feature_records(df_gen, ml_request)
    del df_gen

    # split into train and test
    train_test_split(data_all, ml_request)  # in-place

    # train model
    model_reg = train_regression_model_simple(data_all, ml_request)

# store model
if 1:
    save_reg_model(model_reg, ml_request, config_load)


