"""Train utils"""

from typing import Generator, Tuple, Dict, List, Set
import math

import pandas as pd
import numpy as np

from src.crawler.crawler.utils.mongodb_engine import get_mongodb_records_gen
from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL,
                                           PREFEATURES_TIMESTAMP_COL, TIMESTAMP_FMT, MIN_VID_SAMPS_FOR_DATASET,
                                           NUM_INTVLS_PER_VIDEO, VEC_EMBED_DIMS, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_EMBED_DIM,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom


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
        -> List[Tuple[int]]:
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

    return list(idxs)

def prepare_feature_records(df_gen: Generator[pd.DataFrame, None, None],
                            ml_request: MLRequest) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stream data in and convert to format needed for ML"""
    # define cols to keep
    keys_extract = KEYS_ID + [FEATURES_VECTOR_COL, PREFEATURES_TIMESTAMP_COL] + KEYS_NUM

    # stream all data into RAM
    data_all: List[pd.DataFrame] = []
    while not (df := next(df_gen)).empty:
        df[PREFEATURES_TIMESTAMP_COL] = pd.to_datetime(df[PREFEATURES_TIMESTAMP_COL], format=TIMESTAMP_FMT)
        data_all.append(df[keys_extract])

    df_data = pd.concat(data_all, axis=0, ignore_index=True)

    # filter by group and collect bag-of-words info in a separate DataFrame
    data_all: List[pd.DataFrame] = []
    bows_all: List[pd.DataFrame] = []
    for _, df in df_data.groupby(KEYS_ID):
        # ignore videos without enough measurements
        if len(df) < MIN_VID_SAMPS_FOR_DATASET:
            continue

        # split bow vectors into separate DataFrame
        bows_all.append(df.loc[:0, ['username', 'video_id', FEATURES_VECTOR_COL]])
        data_all.append(df.drop(columns=[FEATURES_VECTOR_COL]))

    df_data = pd.concat(data_all, axis=0, ignore_index=True)
    df_bow = pd.concat(bows_all, axis=0, ignore_index=True)

    # embed bag-of-words features: data-independent dimensionality reduction
    config_ml = ml_request.get_config()
    if config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND:
        model = MLModelLinProjRandom(ml_request)
        model.fit(df_bow) # only uses shape
        df_bow[FEATURES_VECTOR_COL] = model.transform(df_bow, dtype=pd.Series)

    # encode usernames in indicator vectors
    # usernames = df_data['username'].unique()
    # username_code_vecs: Dict[str, List[int]] = {name: [int(i == j) for j in range(len(usernames))]
    #                                             for i, name in enumerate(usernames)}

    # preprocess in groups (one group per video)
    keys_feat = KEYS_NUM + [KEY_TIME_DIFF]

    data_all: List[pd.DataFrame] = []
    for _, df in df_data.groupby(KEYS_ID):
        # add time elapsed since first timestamp
        diffs_ = df[PREFEATURES_TIMESTAMP_COL] - df[PREFEATURES_TIMESTAMP_COL].min()
        df[KEY_TIME_DIFF] = diffs_.dt.total_seconds()
        df = df.drop(columns=[PREFEATURES_TIMESTAMP_COL])

        # generate data samples
        idx_pairs = make_causal_index_pairs(len(df), NUM_INTVLS_PER_VIDEO)
        idxs_src, idxs_tgt = [list(ii) for ii in zip(*idx_pairs)] # e.g. converts [(1, 2), (4, 5), (8, 8)] to [(1, 4, 8), (2, 5, 8)]
        df_src = df.iloc[idxs_src].reset_index(drop=True)
        df_tgt = df[keys_feat].iloc[idxs_tgt].reset_index(drop=True)
        for key in keys_feat:
            df_src[key] = key + '_src'
            df_tgt[key] = key + '_tgt'
        df_feat = pd.concat((df_src, df_tgt), axis=1)

        # add group to dataset
        data_all.append(df_feat)

    df_data = pd.concat(data_all, axis=0, ignore_index=True)

    return df_data, df_bow

def train_test_split(data: pd.DataFrame):
    """Split full dataset into train and test sets"""
    pass

