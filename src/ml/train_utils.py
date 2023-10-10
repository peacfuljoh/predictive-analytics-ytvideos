"""Train utils"""

from typing import Generator, Tuple, Dict, List, Set
import math

import pandas as pd
import numpy as np

from src.crawler.crawler.utils.mongodb_engine import get_mongodb_records_gen
from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL,
                                           PREFEATURES_TIMESTAMP_COL, TIMESTAMP_FMT, MIN_SAMPLES_FOR_DATASET,
                                           NUM_INTVLS_PER_VIDEO)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


KEYS_ID = ['username', 'video_id']
KEYS_NUM = ['comment_count', 'like_count', 'view_count', 'subscriber_count']
KEY_TIME_DIFF = 'time_after_upload' # seconds


""" Load """
def load_feature_records(configs: dict) -> Tuple[Generator[pd.DataFrame, None, None], Dict[str, str]]:
    """Get DataFrame generator for features"""
    assert PREFEATURES_ETL_CONFIG_COL in configs
    assert VOCAB_ETL_CONFIG_COL in configs
    assert FEATURES_ETL_CONFIG_COL in configs

    # get all available config and timestamp combinations
    configs_timestamps = load_config_timestamp_sets_for_features(configs=configs)

    # choose a configs-timestamps combination
    mask = configs_timestamps[FEATURES_TIMESTAMP_COL] == configs_timestamps[FEATURES_TIMESTAMP_COL].max()
    config_chosen = configs_timestamps.loc[mask].iloc[0].to_dict()
    # print_df_full(config_chosen)

    # get a features DataFrame generator
    df_gen = get_mongodb_records_gen(
        DB_FEATURES_NOSQL_DATABASE,
        DB_FEATURES_NOSQL_COLLECTIONS['features'],
        DB_MONGO_CONFIG,
        filter=config_chosen
    )

    return df_gen, {**configs, **config_chosen}


""" Feature preparation """
def make_causal_index_pairs(num_idxs: int,
                            num_pairs: int) \
        -> Set[Tuple[int]]:
    """Make pairs of causal indexes"""
    assert math.factorial(num_pairs) > 100 * num_idxs # ensure plenty of pairs

    num_perms = num_idxs // 2

    idxs = set()
    while len(idxs) < num_pairs:
        idxs_new = np.random.permutation(range(num_idxs))[:2 * num_perms].reshape(num_perms, 2)
        idxs_new = np.sort(idxs_new, axis=1)
        idxs_new = set([tuple(x) for x in idxs_new])
        idxs = idxs.union(idxs_new)
    idxs = idxs[:num_pairs]

    return idxs

def prepare_feature_records(df_gen: Generator[pd.DataFrame, None, None]) -> pd.DataFrame:
    """Stream data in and convert to format needed for ML"""
    # define cols to keep
    keys_extract = KEYS_ID + [FEATURES_VECTOR_COL, PREFEATURES_TIMESTAMP_COL] + KEYS_NUM

    # stream in all data
    data_all: List[pd.DataFrame] = []
    while not (df := next(df_gen)).empty:
        df[PREFEATURES_TIMESTAMP_COL] = pd.to_datetime(df[PREFEATURES_TIMESTAMP_COL], format=TIMESTAMP_FMT)
        data_all.append(df[keys_extract])
    df_data = pd.concat(data_all, axis=0, ignore_index=True)

    """
    inputs: 
        vec
        subscriber count
        username (one-hot-coded vector)
        time_after_upload at source time
        comment/like/view counts at source time
        time_after_upload at target time
    outputs: 
        comment/like/view counts at target time
    """

    # encode usernames
    usernames = df_data['username'].unique()
    username_code_vecs: Dict[str, List[int]] = {name: [int(i == j) for j in range(len(usernames))]
                                                for i, name in enumerate(usernames)}

    # preprocess in groups (one group per video)
    data_all: List[pd.DataFrame] = []
    for _, df in df_data.groupby(KEYS_ID):
        # ignore videos without enough measurements
        if len(df) < MIN_SAMPLES_FOR_DATASET:
            continue

        # add time elapsed since first timestamp
        diffs_ = df[PREFEATURES_TIMESTAMP_COL] - df[PREFEATURES_TIMESTAMP_COL].min()
        df[KEY_TIME_DIFF] = diffs_.dt.total_seconds()

        # generate data samples
        idx_pairs = make_causal_index_pairs(len(df), NUM_INTVLS_PER_VIDEO)

        # TODO: implement the ML feature vector preparation
        #   - how to encode sparse doc vecs?


        # add group to dataset
        data_all.append(df)

    df_data = pd.concat(data_all, axis=0, ignore_index=True)

    return df_data

def train_test_split(data: pd.DataFrame):
    """Split full dataset into train and test sets"""
    pass

