"""Script for testing train and test"""

from src.ml.train_utils import (load_feature_records, train_test_split, prepare_feature_records,
                                train_regression_model_simple, train_regression_model_seq2seq, save_reg_model)
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, VEC_EMBED_DIMS, TRAIN_TEST_SPLIT,
                                           ML_MODEL_TYPE_SEQ2SEQ,
                                           ML_HYPERPARAM_SR_ALPHAS, ML_HYPERPARAM_SR_CV_SPLIT,
                                           ML_HYPERPARAM_SR_CV_COUNT, SPLIT_TRAIN_BY_USERNAME)
from ytpa_utils.misc_utils import print_df_full
from src.ml.ml_request import MLRequest



# specify which version of the feature store to use
preconfig = {
    PREFEATURES_ETL_CONFIG_COL: 'test3',
    VOCAB_ETL_CONFIG_COL: 'test5544',
    FEATURES_ETL_CONFIG_COL: 'test5544',
    # FEATURES_TIMESTAMP_COL: '2023-10-05 17:13:16.668'
}

# specify ML model options
# model_type = ML_MODEL_TYPE_LIN_PROJ_RAND
model_type = ML_MODEL_TYPE_SEQ2SEQ

train_test_split_fract = 0.8 # fraction of data for train
split_train_by_username = True

if model_type == ML_MODEL_TYPE_LIN_PROJ_RAND:
    rlp_density = 0.01 # density of sparse projector matrix
    sr_alphas = [1e-6, 1e-4, 1e-2] # simple regression regularization coefficient
    cv_split = 0.9 # cross-validation split ratio
    cv_count = 10 # number of CV splits

    config_ml = {
        ML_MODEL_TYPE: ML_MODEL_TYPE_SEQ2SEQ,
        ML_MODEL_HYPERPARAMS: {
            ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS, # no. dims in dense vectors after projecting sparse bow vecs
            ML_HYPERPARAM_RLP_DENSITY: rlp_density,
            ML_HYPERPARAM_SR_ALPHAS: sr_alphas,
            ML_HYPERPARAM_SR_CV_SPLIT: cv_split,
            ML_HYPERPARAM_SR_CV_COUNT: cv_count
        }
    }

    split_by = None
elif model_type == ML_MODEL_TYPE_SEQ2SEQ:


    config_ml = {
        ML_MODEL_TYPE: ML_MODEL_TYPE_SEQ2SEQ,
        ML_MODEL_HYPERPARAMS: {
            ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS,
        }
    }

    split_by = 'video_id'
else:
    raise NotImplementedError

config_ml[TRAIN_TEST_SPLIT] = train_test_split_fract
config_ml[SPLIT_TRAIN_BY_USERNAME] = split_train_by_username


ml_request = MLRequest(config_ml)

# see all configs
# from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# get data, train model
if 1:
    # load feature records
    df_gen, config_data = load_feature_records(preconfig, ml_request)

    # preprocess feature records
    data_all, model_embed = prepare_feature_records(df_gen, ml_request)

    # split into train and test
    train_test_split(data_all, ml_request, split_by=split_by)  # in-place

    # train model
    if model_type == ML_MODEL_TYPE_LIN_PROJ_RAND:
        model_reg = train_regression_model_simple(data_all, ml_request)
    elif model_type == ML_MODEL_TYPE_SEQ2SEQ:
        model_reg = train_regression_model_seq2seq(data_all, ml_request)

# store model
if 0:
    save_reg_model(model_reg, ml_request, preconfig)


