"""Script for testing train and test"""

import math

from src.ml.train_utils import load_feature_records, train_test_split, prepare_feature_records
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, VEC_EMBED_DIMS)
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
n_features_approx = 30000
rlp_density = 1 / math.sqrt(n_features_approx)

config_ml = {
    ML_MODEL_TYPE: ML_MODEL_TYPE_LIN_PROJ_RAND,
    ML_MODEL_HYPERPARAMS: {
        ML_HYPERPARAM_EMBED_DIM: VEC_EMBED_DIMS, # number of dimensions in dense vectors after projecting sparse bow vecs
        ML_HYPERPARAM_RLP_DENSITY: rlp_density # density of sparse projector matrix
    }
}

ml_request = MLRequest(config_ml)

# see all configs
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# load feature records
df_gen, config_data = load_feature_records(config_load)

# preprocess feature records
data_all = prepare_feature_records(df_gen, ml_request)

# split into train and test
data_split = train_test_split(data_all)

# train model


# evaluate on held-out test set

