"""Train utils"""

from typing import Generator, Tuple, Dict, List, Optional

import pandas as pd
import numpy as np

from src.crawler.crawler.utils.mongodb_engine import get_mongodb_records_gen
from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL,
                                           PREFEATURES_TIMESTAMP_COL, TIMESTAMP_FMT, MIN_VID_SAMPS_FOR_DATASET,
                                           NUM_INTVLS_PER_VIDEO, ML_MODEL_TYPE, ML_MODEL_TYPE_LIN_PROJ_RAND, TRAIN_TEST_SPLIT,
                                           KEYS_TRAIN_ID, KEYS_TRAIN_NUM, KEYS_TRAIN_NUM_TGT,
                                           KEY_TRAIN_TIME_DIFF)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import load_config_timestamp_sets_for_features
from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom, MLModelRegressionSimple


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']



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
        -> Tuple[List[int], List[int]]:
    """Make pairs of causal indexes"""
    # assert math.factorial(num_pairs) > 100 * num_idxs # ensure plenty of pairs

    num_proposals = 5

    idxs = [[], []] # src and tgt indices
    for i in np.random.permutation(range(num_idxs)):
        jj_new = np.random.permutation(range(i + 1, num_idxs))[:num_proposals]
        idxs[0] += [i] * len(jj_new)
        idxs[1] += list(jj_new)
        if len(idxs[0]) >= num_pairs:
            while len(idxs[0]) > num_pairs:
                idxs[0].pop()
                idxs[1].pop()
            break
    return idxs[0], idxs[1]

def embed_bow_with_lin_proj_rand(ml_request: MLRequest,
                                 df_bow: pd.DataFrame) \
        -> MLModelLinProjRandom:
    # embed bag-of-words features: data-independent dimensionality reduction
    model = MLModelLinProjRandom(ml_request)
    model.fit(df_bow) # only uses shape
    df_bow[FEATURES_VECTOR_COL] = model.transform(df_bow, dtype=pd.Series)
    return model

def stream_all_features_into_ram(df_gen: Generator[pd.DataFrame, None, None],
                                 keys_extract: Optional[List[str]] = None) \
        -> pd.DataFrame:
    # stream all feature DataFrames into RAM
    data_all: List[pd.DataFrame] = []

    while not (df := next(df_gen)).empty:
        df[PREFEATURES_TIMESTAMP_COL] = pd.to_datetime(df[PREFEATURES_TIMESTAMP_COL], format=TIMESTAMP_FMT)
        if keys_extract is not None:
            df = df[keys_extract]
        data_all.append(df)

    return pd.concat(data_all, axis=0, ignore_index=True)

def split_feature_df_and_filter(df_data: pd.DataFrame,
                                min_samps: Optional[int] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    # filter by group and collect bag-of-words info in a separate DataFrame
    data_all: List[pd.DataFrame] = []
    bows_all: List[pd.DataFrame] = []

    for _, df in df_data.groupby(KEYS_TRAIN_ID):
        # ignore videos without enough measurements
        if min_samps is not None and len(df) < min_samps:
            continue

        # split bow vectors into separate DataFrame
        assert len(df[FEATURES_VECTOR_COL].drop_duplicates()) == 1 # ensure unique bow representation for this group
        bows_all.append(df.iloc[:1][KEYS_TRAIN_ID + [FEATURES_VECTOR_COL]])
        data_all.append(df.drop(columns=[FEATURES_VECTOR_COL]))

    df_data = pd.concat(data_all, axis=0, ignore_index=True)
    df_bow = pd.concat(bows_all, axis=0, ignore_index=True)

    return df_data, df_bow

def prepare_input_output_vectors(df_data: pd.DataFrame,
                                 keys_feat_src: List[str],
                                 keys_feat_tgt: List[str]) \
        -> pd.DataFrame:
    """
    Prepare non-bow features. Forms measurement pairs (in time).
    """
    data_all: List[pd.DataFrame] = []

    for ids, df in df_data.groupby(KEYS_TRAIN_ID): # one group per video
        # sort by timestamp (index pairing assumes time-ordering)
        df = df.sort_values(by=[PREFEATURES_TIMESTAMP_COL])

        # add time elapsed since first timestamp
        diffs_ = df[PREFEATURES_TIMESTAMP_COL] - df[PREFEATURES_TIMESTAMP_COL].min()
        df[KEY_TRAIN_TIME_DIFF] = diffs_.dt.total_seconds()
        df = df.drop(columns=[PREFEATURES_TIMESTAMP_COL])  # drop timestamp

        # generate data samples by causal temporal pairs
        idxs_src, idxs_tgt = make_causal_index_pairs(len(df), NUM_INTVLS_PER_VIDEO)
        df_src = df.iloc[idxs_src].reset_index(drop=True)  # has video identifiers
        df_tgt = df[keys_feat_tgt].iloc[idxs_tgt].reset_index(drop=True) # does not have video identifiers (otherwise concat would duplicate)
        df_src = df_src.rename(columns={key: key + '_src' for key in keys_feat_src})
        df_tgt = df_tgt.rename(columns={key: key + '_tgt' for key in keys_feat_tgt})
        df_feat = pd.concat((df_src, df_tgt), axis=1)

        # double-check time ordering
        assert all(df_feat[KEY_TRAIN_TIME_DIFF + '_tgt'] > df_feat[KEY_TRAIN_TIME_DIFF + '_src'])

        # add group to dataset
        data_all.append(df_feat)

    df_data = pd.concat(data_all, axis=0, ignore_index=True)

    return df_data

def prepare_feature_records(df_gen: Generator[pd.DataFrame, None, None],
                            ml_request: MLRequest) \
        -> Tuple[Dict[str, pd.DataFrame], MLModelLinProjRandom]:
    """Stream data in and convert to format needed for ML"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

    # setup
    keys_extract = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, PREFEATURES_TIMESTAMP_COL] + KEYS_TRAIN_NUM  # define cols to keep
    keys_feat_src = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
    keys_feat_tgt = KEYS_TRAIN_NUM_TGT + [KEY_TRAIN_TIME_DIFF] # columns of interest for output vectors

    # stream all data into RAM
    df_data = stream_all_features_into_ram(df_gen, keys_extract)

    # filter by group and collect bag-of-words info in a separate DataFrame
    df_nonbow, df_bow = split_feature_df_and_filter(df_data, MIN_VID_SAMPS_FOR_DATASET)

    # embed bag-of-words features: data-independent dimensionality reduction
    model_embed = embed_bow_with_lin_proj_rand(ml_request, df_bow) # df_bow modified in-place

    # encode usernames in indicator vectors
    # usernames = df_data['username'].unique()
    # username_code_vecs: Dict[str, List[int]] = {name: [int(i == j) for j in range(len(usernames))]
    #                                             for i, name in enumerate(usernames)}

    # prepare non-bow features
    df_nonbow = prepare_input_output_vectors(df_nonbow, keys_feat_src, keys_feat_tgt)

    return dict(nonbow=df_nonbow, bow=df_bow), model_embed

def train_test_split(data: Dict[str, pd.DataFrame],
                     ml_request: MLRequest):
    """Split full dataset into train and test sets"""
    tt_split: float = ml_request.get_config()[TRAIN_TEST_SPLIT]

    data_nonbow = data['nonbow']
    num_samps = len(data_nonbow)
    num_samp_train = int(num_samps * tt_split)
    ii = np.random.permutation(num_samps)
    data_train = data_nonbow.iloc[ii[:num_samp_train]]
    data_test = data_nonbow.iloc[ii[num_samp_train:]]

    data['nonbow_train'] = data_train
    data['nonbow_test'] = data_test
    del data['nonbow'] # save on memory

def train_regression_model_simple(data: Dict[str, pd.DataFrame],
                                  ml_request: MLRequest):
    """Train simple regression model"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

    # fit model
    model_reg = MLModelRegressionSimple(ml_request, verbose=True)
    model_reg.fit(data['nonbow_train'], data['bow'])

    # predict
    out = model_reg.predict(data['nonbow_test'])

    # see predictions
    if 1:
        import matplotlib.pyplot as plt

        n_plots = 5
        n_pred = 50

        fig = plt.subplots(n_plots, 1, sharex=True)

        for i, row in data['bow'].iterrows():
            if i < n_plots:
                # get test data
                username = row['username']
                video_id = row['video_id']

                dd = data['nonbow_train']
                dd = dd[(dd['username'] == username) * (dd['video_id'] == video_id)]

                t0_col = 'time_after_upload_src'
                tf_col = 'time_after_upload_tgt'

                rec_first = dd.loc[dd[t0_col].argmin()]
                rec_last = dd.loc[dd[t0_col].argmax()]
                df = pd.DataFrame([rec_first for _ in range(n_pred)])
                df[tf_col] = np.linspace(rec_first[t0_col], rec_last[t0_col], n_pred)

                # predict with model
                

                # plot it


                a = 5

        plt.show()


    return model_reg