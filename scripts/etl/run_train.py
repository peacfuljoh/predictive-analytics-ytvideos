"""Script for testing train and test"""

from src.ml.train_utils import load_feature_records
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                                           FEATURES_TIMESTAMP_COL)
from src.crawler.crawler.utils.misc_utils import print_df_full



# specify which version of the feature store to use
configs = {
    PREFEATURES_ETL_CONFIG_COL: 'test3',
    VOCAB_ETL_CONFIG_COL: 'test007',
    FEATURES_ETL_CONFIG_COL: 'test007',
    # FEATURES_TIMESTAMP_COL: '2023-10-05 17:13:16.668'
}
# configs = None

# see all configs
# configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)
# print_df_full(configs_timestamps)

# load feature records
df_gen = load_feature_records(configs)

# split into train and test
# TODO: implement next

# train model


# evaluate on held-out test set

