"""Script for running model train and test"""

import os

from ytpa_utils.misc_utils import print_df_full
from ytpa_utils.io_utils import load_pickle, save_pickle

from src.ml.train_utils import (load_feature_records, train_test_split, prepare_feature_records,
                                train_regression_model_simple, train_regression_model_seq2seq, save_reg_model,
                                print_train_data_stats, predict_seq2seq)
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, VEC_EMBED_DIMS, TRAIN_TEST_SPLIT,
                                           ML_MODEL_TYPE_SEQ2SEQ,
                                           ML_HYPERPARAM_SR_ALPHAS, ML_HYPERPARAM_SR_CV_SPLIT,
                                           ML_HYPERPARAM_SR_CV_COUNT, SPLIT_TRAIN_BY_USERNAME,
                                           MODEL_ID)
from src.ml.ml_request import MLRequest



USE_LOCAL_DATA_CACHE = True # very dumb cache...be careful
CACHE_FNAME = '/home/nuc/Desktop/temp/ytpa_data_for_ml.pickle'

# os.remove(CACHE_FNAME) # delete cache file

run_train = True
run_predict = False



# specify which version of the feature store to use
preconfig = {
    PREFEATURES_ETL_CONFIG_COL: 'test3',
    VOCAB_ETL_CONFIG_COL: 'test4422', # same as features config name
    FEATURES_ETL_CONFIG_COL: 'test4422',
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
    rlp_density = 0.01  # density of sparse projector matrix

    config_ml = {
        ML_MODEL_TYPE: ML_MODEL_TYPE_SEQ2SEQ,
        ML_MODEL_HYPERPARAMS: {
            ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS,
            ML_HYPERPARAM_RLP_DENSITY: rlp_density
        }
    }

    split_by = 'video_id'
else:
    raise NotImplementedError

# additional config options
config_ml[TRAIN_TEST_SPLIT] = train_test_split_fract
config_ml[SPLIT_TRAIN_BY_USERNAME] = split_train_by_username
config_ml[MODEL_ID] = [
    '2024-02-06_15-45-14',
    '2024-02-06_17-19-58',
    '2024-02-06_15-45-30' # sub-0.1 test loss
][3] # CNN only, sub in embeds vec, LSTMs with varying numbers of 2-layer blocks (1, 3, 6)
config_ml[MODEL_ID] = [
    '2024-02-08_13-55-57'
][0]


# create ML request object (includes field validation)
ml_request = MLRequest(config_ml)

# see all configs
# from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# get data, train model
if 1:
    # get data
    if USE_LOCAL_DATA_CACHE and os.path.exists(CACHE_FNAME):
        data_all = load_pickle(CACHE_FNAME)
    else:
        df_gen, config_data = load_feature_records(preconfig, ml_request)
        data_all, model_embed = prepare_feature_records(df_gen, ml_request)
        train_test_split(data_all, ml_request, split_by=split_by)  # in-place
        if USE_LOCAL_DATA_CACHE:
            save_pickle(CACHE_FNAME, data_all)

    # show training data stats
    print_train_data_stats(data_all, ml_request)

    # choose subset of data
    if 1:
        data_all['nonbow_train'] = data_all['nonbow_train'].query(f"username == 'CNN'")

    # train model
    if run_train:
        if model_type == ML_MODEL_TYPE_LIN_PROJ_RAND:
            model_reg = train_regression_model_simple(data_all, ml_request)
        elif model_type == ML_MODEL_TYPE_SEQ2SEQ:
            model_reg = train_regression_model_seq2seq(data_all, ml_request)

    # predict and visualize
    if run_predict:
        if model_type == ML_MODEL_TYPE_SEQ2SEQ:
            predict_seq2seq(data_all, ml_request)

# store model
if 0:
    save_reg_model(model_reg, ml_request, preconfig)

