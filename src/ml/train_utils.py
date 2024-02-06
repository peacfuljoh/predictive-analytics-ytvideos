"""Train utils"""

import datetime
import os
from typing import Generator, Tuple, Dict, List, Optional, Union
import requests
from pprint import pprint

import pandas as pd
import numpy as np

from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL, TIMESTAMP_FMT,
                                           MIN_VID_SAMPS_FOR_DATASET, MIN_VID_SAMPS_FOR_DATASET_SEQ2SEQ,
                                           NUM_INTVLS_PER_VIDEO, ML_MODEL_TYPE, ML_MODEL_TYPE_LIN_PROJ_RAND,
                                           ML_MODEL_TYPE_SEQ2SEQ, ML_MODEL_TYPES, KEYS_TRAIN_NUM,
                                           TRAIN_TEST_SPLIT, KEYS_TRAIN_ID, KEYS_TRAIN_NUM_TGT,
                                           KEY_TRAIN_TIME_DIFF, SPLIT_TRAIN_BY_USERNAME, KEYS_FOR_PRED_NONBOW_ID,
                                           COL_VIDEO_ID, COL_USERNAME, COL_TIMESTAMP_ACCESSED,
                                           MODEL_MODEL_OBJ, MODEL_META_ID, MODEL_SPLIT_NAME,
                                           TIMESTAMP_CONVERSION_FMTS_ENCODE, TIMESTAMP_CONVERSION_FMTS_DECODE,
                                           COL_VIEW_COUNT, COL_LIKE_COUNT, TRAIN_SEQ_PERIOD, MODEL_ID,
                                           COL_SEQ_SPLIT_FOR_PRED, SEQ_SPLIT_INCLUDE, SEQ_SPLIT_EXCLUDE,
                                           SEQ_SPLIT_PREDICT)
from src.crawler.crawler.config import FEATURES_ENDPOINT, CONFIG_TIMESTAMP_SETS_ENDPOINT, MODEL_ROOT
from src.etl.etl_utils import convert_ts_fmt_for_mongo_id
from ytpa_utils.val_utils import is_dict_of_instances, is_subset
from ytpa_utils.time_utils import get_ts_now_str
from ytpa_utils.df_utils import df_dt_codec, resample_one_df_in_time
from ytpa_utils.misc_utils import print_df_full
from ytpa_api_utils.websocket_utils import df_generator_ws
from src.ml.ml_constants import KEYS_EXTRACT_LIN, KEYS_FEAT_SRC_LIN, KEYS_FEAT_TGT_LIN, KEYS_EXTRACT_SEQ2SEQ
from src.ml.ml_request import MLRequest
from src.ml.model_utils import load_models
from src.ml.models_linear import MLModelBaseRegressionSimple
from src.ml.models_nn import MLModelSeq2Seq, _predict
from src.ml.models_projection import MLModelLinProjRandom
from src.schemas.schema_validation import validate_mongodb_records_schema
from src.schemas.schemas import SCHEMAS_MONGODB





""" Load """
def load_feature_records(configs: dict,
                         ml_request: MLRequest) \
        -> Tuple[Generator[pd.DataFrame, None, None], Dict[str, str]]:
    """Get DataFrame generator for features"""
    print('\nLoading feature records')

    assert PREFEATURES_ETL_CONFIG_COL in configs
    assert VOCAB_ETL_CONFIG_COL in configs
    assert FEATURES_ETL_CONFIG_COL in configs

    # get all available config and timestamp combinations
    res = requests.post(CONFIG_TIMESTAMP_SETS_ENDPOINT, json=configs)
    configs_timestamps = pd.DataFrame(res.json())
    df_dt_codec(configs_timestamps, TIMESTAMP_CONVERSION_FMTS_DECODE)

    # choose a configs-timestamps combination
    mask = configs_timestamps[FEATURES_TIMESTAMP_COL] == configs_timestamps[FEATURES_TIMESTAMP_COL].max()
    config_chosen: pd.DataFrame = configs_timestamps.loc[mask].iloc[:1]
    # print_df_full(config_chosen)

    # get a features DataFrame generator
    config_chosen_strs = config_chosen.copy()
    df_dt_codec(config_chosen_strs, TIMESTAMP_CONVERSION_FMTS_ENCODE) # make msg serializable
    filter_: dict = config_chosen_strs.iloc[0].to_dict()
    etl_config_options = {'extract': {'filter': filter_}}
    df_gen = df_generator_ws(FEATURES_ENDPOINT, etl_config_options, transformations=TIMESTAMP_CONVERSION_FMTS_DECODE)

    return df_gen, {**configs, **config_chosen.iloc[0].to_dict()}



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
    """Embed bag-of-words features: raw_data-independent dimensionality reduction"""
    model = MLModelLinProjRandom(ml_request)
    model.fit(df_bow) # only uses shape
    df_bow[FEATURES_VECTOR_COL] = model.transform(df_bow, dtype=pd.Series)
    return model

def stream_all_features_into_ram(df_gen: Generator[pd.DataFrame, None, None],
                                 keys_extract: Optional[List[str]] = None) \
        -> pd.DataFrame:
    """Stream all feature DataFrames into RAM"""
    data_all: List[pd.DataFrame] = []

    # while not (df := next(df_gen)).empty:
    for df in df_gen:
        df[COL_TIMESTAMP_ACCESSED] = pd.to_datetime(df[COL_TIMESTAMP_ACCESSED], format=TIMESTAMP_FMT)
        if keys_extract is not None:
            df = df[keys_extract]
        data_all.append(df)

    return pd.concat(data_all, axis=0, ignore_index=True)

def split_feature_df_and_filter(df_data: pd.DataFrame,
                                min_samps: Optional[int] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split feature DataFrame (with filtering) into non-bow and bow."""
    # filter by group and collect bag-of-words info in a separate DataFrame
    nonbow_all: List[pd.DataFrame] = []
    bows_all: List[pd.DataFrame] = []

    for _, df in df_data.groupby(KEYS_TRAIN_ID):
        # ignore videos without enough measurements
        if min_samps is not None and len(df) < min_samps:
            continue

        # split bow vectors into separate DataFrame
        assert len(df[FEATURES_VECTOR_COL].drop_duplicates()) == 1 # ensure unique bow representation for this group
        bows_all.append(df.iloc[:1][KEYS_TRAIN_ID + [FEATURES_VECTOR_COL]])
        nonbow_all.append(df.drop(columns=[FEATURES_VECTOR_COL]))

    df_nonbow = pd.concat(nonbow_all, axis=0, ignore_index=True)
    df_bow = pd.concat(bows_all, axis=0, ignore_index=True)

    return df_nonbow, df_bow

def prepare_input_output_vectors(df_data: pd.DataFrame,
                                 keys_feat_src: List[str],
                                 keys_feat_tgt: List[str]) \
        -> pd.DataFrame:
    """
    Prepare non-bow features. Forms measurement pairs (in time).
    """
    data_all: List[pd.DataFrame] = []

    for _, df in df_data.groupby(KEYS_TRAIN_ID): # one group per video
        # sort by timestamp (index pairing assumes time-ordering)
        df = df.sort_values(by=[COL_TIMESTAMP_ACCESSED])

        # add time elapsed since first timestamp
        diffs_ = df[COL_TIMESTAMP_ACCESSED] - df[COL_TIMESTAMP_ACCESSED].min()
        df[KEY_TRAIN_TIME_DIFF] = diffs_.dt.total_seconds()
        df = df.drop(columns=[COL_TIMESTAMP_ACCESSED])  # drop timestamp

        # generate raw_data samples by causal temporal pairs
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

def resample_to_uniform_grid(df_all: pd.DataFrame,
                             period: int) \
        -> pd.DataFrame:
    """
    Resample stats sequences in dataframe to uniform time spacing.
    Input arg 'period' is in units of seconds.

    Note: when making changes to this function, ensure that ordering of numerical columns remains the same throughout.
    """
    MAX_TS_GAP = 12 * 3600  # max gap in time before linear interpolation is used

    # validate columns and types
    cols_non_num = KEYS_TRAIN_ID + [COL_TIMESTAMP_ACCESSED]
    assert is_subset(cols_non_num, df_all.columns)
    cols_num = list(set(df_all.columns) - set(cols_non_num)) # WARNING: use this ordering of numerical cols throughout
    assert all([dtype in ['int32', 'int64', 'float32', 'float64']
                for col, dtype in zip(df_all.columns, df_all.dtypes) if col in cols_num])

    # resample numerical data in groups (one group per video)
    df_resamp_all = []
    for ids, df in df_all.groupby(KEYS_TRAIN_ID):
        # if ids[0] == 'FoxNews' and ids[1] == '48YaHEBPXQc':
        #     a = 5
        df_resamp = resample_one_df_in_time(df, period, KEYS_TRAIN_ID, cols_num, COL_TIMESTAMP_ACCESSED,
                                            max_ts_gap=MAX_TS_GAP, omit_zeroes=True)
        df_resamp_all.append(df_resamp)

    return pd.concat(df_resamp_all, axis=0, ignore_index=True)

def prepare_feature_records_lin_reg(df_gen: Generator[pd.DataFrame, None, None],
                                    ml_request: MLRequest) \
        -> Tuple[Dict[str, pd.DataFrame], MLModelLinProjRandom]:
    """Prepare feature records for linear regression model."""
    # stream all raw_data into RAM
    df_data = stream_all_features_into_ram(df_gen, KEYS_EXTRACT_LIN)

    # filter by group and collect bag-of-words info in a separate DataFrame
    df_nonbow, df_bow = split_feature_df_and_filter(df_data, MIN_VID_SAMPS_FOR_DATASET)

    # embed bag-of-words features: raw_data-independent dimensionality reduction
    model_embed = embed_bow_with_lin_proj_rand(ml_request, df_bow)  # df_bow modified in-place

    # prepare non-bow features
    df_nonbow = prepare_input_output_vectors(df_nonbow, KEYS_FEAT_SRC_LIN, KEYS_FEAT_TGT_LIN)

    return dict(nonbow=df_nonbow, bow=df_bow), model_embed

def prepare_feature_records_seq2seq(df_gen: Generator[pd.DataFrame, None, None],
                                    ml_request: MLRequest) \
        -> Tuple[Dict[str, pd.DataFrame], MLModelLinProjRandom]:
    """Prepare feature records for sequence-to-sequence model."""
    show = False # debug

    # stream all raw_data into RAM
    df_data = stream_all_features_into_ram(df_gen, KEYS_EXTRACT_SEQ2SEQ)

    # filter by group and collect bag-of-words info in a separate DataFrame
    df_nonbow, df_bow = split_feature_df_and_filter(df_data, MIN_VID_SAMPS_FOR_DATASET)

    # embed bag-of-words features: data-independent dimensionality reduction
    model_embed = embed_bow_with_lin_proj_rand(ml_request, df_bow)  # df_bow modified in-place

    # resample to uniform grid
    if show:
        df_nonbow_orig = df_nonbow.copy()
    df_nonbow = resample_to_uniform_grid(df_nonbow, TRAIN_SEQ_PERIOD)

    # filter by sequence length after resampling
    df_nonbow = filter_by_seq_len(df_nonbow, MIN_VID_SAMPS_FOR_DATASET_SEQ2SEQ)

    # visualize (debug)
    if show:
        import matplotlib.pyplot as plt

        for ids, df_nb_or in df_nonbow_orig.groupby(KEYS_TRAIN_ID):
            if ids[0] in []:#'CNN', 'FoxNews', 'NBCNews', 'TheYoungTurks']:
                continue
            video_id_ = ids[1]
            df_nb = df_nonbow[df_nonbow[COL_VIDEO_ID] == video_id_]

            fig, axes = plt.subplots(4, 1, figsize=(10, 8))
            for i, key in enumerate(KEYS_TRAIN_NUM):
                axes[i].scatter(df_nb_or[COL_TIMESTAMP_ACCESSED], df_nb_or[key], c='k', s=5)
                axes[i].scatter(df_nb[COL_TIMESTAMP_ACCESSED], df_nb[key], facecolor='none', edgecolor='r', s=20)
                if i == 0:
                    axes[i].set_title(str(ids))
                if key in [COL_VIEW_COUNT, COL_LIKE_COUNT]:
                    axes[i].set_ylim([0, 1.05 * df_nb_or[key].max()])
                axes[i].set_ylabel(key)

            plt.show()

    return dict(nonbow=df_nonbow, bow=df_bow), model_embed

def filter_by_seq_len(df: pd.DataFrame,
                      min_seq_len: int) \
        -> pd.DataFrame:
    """Filter DataFrame (nonbow) to have minimum sequence length per video"""
    seq_lens: pd.Series = df.groupby(COL_VIDEO_ID).size()
    video_ids_to_keep: List[str] = list(seq_lens[seq_lens >= min_seq_len].index)

    return df[df[COL_VIDEO_ID].isin(video_ids_to_keep)]

def prepare_feature_records(df_gen: Generator[pd.DataFrame, None, None],
                            ml_request: MLRequest) \
        -> Tuple[Dict[str, pd.DataFrame], MLModelLinProjRandom]:
    """Stream raw_data in and convert to format needed for ML"""
    print('\nPreparing feature records')

    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] in ML_MODEL_TYPES

    if config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND:
        data_all, model_embed = prepare_feature_records_lin_reg(df_gen, ml_request)
    elif config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_SEQ2SEQ:
        data_all, model_embed = prepare_feature_records_seq2seq(df_gen, ml_request)
    else:
        raise NotImplementedError

    return data_all, model_embed






""" Train-test split """
def train_test_split_uniform(data: Dict[str, pd.DataFrame],
                             tt_split: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train-test split uniformly across all samples"""
    data_nonbow = data['nonbow']
    num_samps = len(data_nonbow)
    num_train = int(num_samps * tt_split)
    ii = np.random.permutation(num_samps)
    data_train = data_nonbow.iloc[ii[:num_train]]
    data_test = data_nonbow.iloc[ii[num_train:]]

    return data_train, data_test

def train_test_split_video_id(data: Dict[str, pd.DataFrame],
                              tt_split: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train-test split by video_id. Split is performed one a per-username basis and then merged across users."""
    data_train = []
    data_test = []
    for _, df in data['nonbow'].groupby(COL_USERNAME):
        video_ids = df[COL_VIDEO_ID].unique()
        num_vids = len(video_ids)
        num_train = int(num_vids * tt_split)
        ii = np.random.permutation(num_vids)
        video_ids_train = video_ids[ii[:num_train]]
        video_ids_test = video_ids[ii[num_train:]]
        mask_train = df[COL_VIDEO_ID].isin(video_ids_train)
        mask_test = df[COL_VIDEO_ID].isin(video_ids_test)
        data_train.append(df[mask_train])
        data_test.append(df[mask_test])

    data_train = pd.concat(data_train, axis=0, ignore_index=True)
    data_test = pd.concat(data_test, axis=0, ignore_index=True)

    return data_train, data_test

def train_test_split(data: Dict[str, pd.DataFrame],
                     ml_request: MLRequest,
                     split_by: Optional[str] = None):
    """
    Split non-bow data into train and test sets.
    Input arg 'split_by' determines how examples are split (i.e. uniformly, by video_id, etc.).
    """
    print('\nApplying train-test split')

    assert split_by in [None, 'video_id']

    tt_split: float = ml_request.get_config()[TRAIN_TEST_SPLIT]

    if split_by is None:
        data['nonbow_train'], data['nonbow_test'] = train_test_split_uniform(data, tt_split)
    elif split_by == 'video_id':
        data['nonbow_train'], data['nonbow_test'] = train_test_split_video_id(data, tt_split)

    del data['nonbow'] # save on memory




""" Train regression models """
def train_regression_model_simple(data: Dict[str, pd.DataFrame],
                                  ml_request: MLRequest) \
        -> Union[MLModelBaseRegressionSimple, Dict[str, MLModelBaseRegressionSimple]]:
    """Train simple regression model"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

    # fit model
    if not config_ml[SPLIT_TRAIN_BY_USERNAME]:
        model = MLModelBaseRegressionSimple(ml_request, verbose=True)
        model.fit(data['nonbow_train'], data['bow'])
    else:
        usernames = data['nonbow_train'][COL_USERNAME].drop_duplicates()
        model = {uname: MLModelBaseRegressionSimple(ml_request, verbose=True) for uname in usernames}
        for uname, model_ in model.items():
            df_nonbow = data['nonbow_train']
            df_nonbow = df_nonbow[df_nonbow[COL_USERNAME] == uname]
            df_bow = data['bow']
            df_bow = df_bow[df_bow[COL_USERNAME] == uname]
            model_.fit(df_nonbow, df_bow)

        if 0:
            # try model encoding and decoding
            model_dicts = {}
            for uname, model_ in model.items():
                model_dicts[uname] = model_.encode()
            model = {}
            for uname in usernames:
                model[uname] = MLModelBaseRegressionSimple(verbose=True, model_dict=model_dicts[uname])

    # see predictions
    if 1:
        import matplotlib.pyplot as plt

        if not config_ml[SPLIT_TRAIN_BY_USERNAME]:
            plot_predictions(data, model)
        else:
            for uname, model_ in model.items():
                df_nonbow = data['nonbow_test']
                data_ = dict(nonbow_test=df_nonbow[df_nonbow[COL_USERNAME] == uname])
                plot_predictions(data_, model_)

        plt.show()

    return model

def train_regression_model_seq2seq(data: Dict[str, pd.DataFrame],
                                   ml_request: MLRequest) \
        -> MLModelSeq2Seq:
    """Train sequence-to-sequence prediction model"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_SEQ2SEQ

    # create new directory for models in this run
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)
    dt_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = os.path.join(MODEL_ROOT, dt_str)
    os.makedirs(model_dir)

    # training data overview
    df_overview = data['nonbow_train'].groupby([COL_USERNAME, COL_VIDEO_ID]).size().to_frame('num_frames').reset_index()
    # df_overview = df_overview.reset_index()#.rename(columns={0: 'num_frames'})
    groupby = df_overview.drop(columns=COL_VIDEO_ID).groupby(COL_USERNAME)
    df_overview_counts = groupby.sum()
    df_overview_counts['num_videos'] = groupby.size()

    print(f'\n== Training seq2seq models for individual users ==')
    print_df_full(df_overview_counts)

    # fit model
    if not config_ml[SPLIT_TRAIN_BY_USERNAME]:
        metadata = dict(
            usernames=data['nonbow_train'][COL_USERNAME].unique().tolist(),
            video_ids=data['nonbow_train'][COL_VIDEO_ID].unique().tolist()
        )
        model_c = MLModelSeq2Seq(ml_request, verbose=True, model_dir=model_dir, metadata=metadata)
        model_c.fit(data['nonbow_train'], data['bow'])
    else:
        usernames = data['nonbow_train'][COL_USERNAME].drop_duplicates()
        model_c = {}
        for uname in usernames:
            df_nonbow = data['nonbow_train']
            df_nonbow = df_nonbow[df_nonbow[COL_USERNAME] == uname]
            df_bow = data['bow']
            df_bow = df_bow[df_bow[COL_USERNAME] == uname]

            metadata = dict(
                usernames=df_nonbow[COL_USERNAME].unique().tolist(),
                video_ids=df_nonbow[COL_VIDEO_ID].unique().tolist()
            )

            df_nonbow_test = data['nonbow_test']
            df_nonbow_test = df_nonbow_test[df_nonbow_test[COL_USERNAME] == uname]

            print(f'\n== Training seq2seq model for user {uname} ==')
            print(f'  {df_overview_counts.loc[uname]["num_videos"]} videos, '
                  f'{df_overview_counts.loc[uname]["num_frames"]} frames')

            model_c[uname] = MLModelSeq2Seq(ml_request, verbose=True, model_dir=model_dir, metadata=metadata)
            model_c[uname].fit(df_nonbow, df_bow, data_nonbow_test=df_nonbow_test)

    # see predictions
    if 0:
        import matplotlib.pyplot as plt

        # if not config_ml[SPLIT_TRAIN_BY_USERNAME]:
        #     plot_predictions(data, model)
        # else:
        #     for uname, model_ in model.items():
        #         df_nonbow = data['nonbow_test']
        #         data_ = dict(nonbow_test=df_nonbow[df_nonbow[COL_USERNAME] == uname])
        #         plot_predictions(data_, model_)

        plt.show()

    return model_c

def predict_seq2seq(data: Dict[str, pd.DataFrame],
                    ml_request: MLRequest):
    """Perform sequential predictions with a desired model."""
    print(f'\n\n ===== Running prediction for seq2seq model =====\n')

    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_SEQ2SEQ
    assert MODEL_ID in config_ml

    # load model
    models_info = load_models(ml_request)
    model_ids = list(models_info.keys())

    # print useful info
    if 1:
        # dict_keys(['model', 'stats', 'modules', 'options', 'metadata', 'epoch'])
        for model_id in model_ids:
            model_ = models_info[model_id]
            print(f'== Trained model info ==')
            print(f"  Epoch: {model_['epoch']}")
            print(f"  Usernames: {model_['metadata']['usernames']}")
            print(f"  Number of videos: {len(model_['metadata']['video_ids'])}")

    # make sequence splits (decide where to predict from in each sequence)
    make_seq_splits_for_predict(data)

    # predict
    df_preds = make_seq2seq_predictions_with_models(data, models_info)

    # see predictions
    if 1:
        show_seq2seq_predictions(data, df_preds)

def make_seq_splits_for_predict(data: Dict[str, pd.DataFrame]):
    """Make decisions about where to split test sequences into input and to-predict. In-place modification."""
    split_flags = pd.Series(index=data['nonbow_test'].index)
    split_flags[:] = SEQ_SPLIT_EXCLUDE
    for video_id, df_ in data['nonbow_test'].groupby(COL_VIDEO_ID):
        idx_split = int(0.5 * len(df_))
        split_flags[df_.index[:idx_split]] = SEQ_SPLIT_INCLUDE
    data['nonbow_test'][COL_SEQ_SPLIT_FOR_PRED] = split_flags

def make_seq2seq_predictions_with_models(data: Dict[str, pd.DataFrame],
                                         models_info: Dict[str, dict]) \
        -> pd.DataFrame:
    """Make seq2seq predictions with loaded models"""
    model_ids = list(models_info.keys())

    pred_opts = {}

    dfs = []
    for model_id in model_ids:
        model_info_ = models_info[model_id]

        unames = model_info_['metadata']['usernames']

        model = model_info_['model']

        query = f"({COL_USERNAME} in {unames}) & ({COL_SEQ_SPLIT_FOR_PRED} == '{SEQ_SPLIT_INCLUDE}')"
        df_nonbow_input = data['nonbow_test'].query(query).drop(columns=[COL_SEQ_SPLIT_FOR_PRED])

        df_pred_ = _predict(df_nonbow_input, data['bow'], pred_opts=pred_opts, model=model)
        dfs.append(df_pred_)

    df_pred_all = pd.concat(dfs, axis=0, ignore_index=True)

    return df_pred_all








""" Show and tell """
def plot_predictions(data: Dict[str, pd.DataFrame],
                     model):
    """Plot test raw_data and predictions for various src times."""
    import matplotlib.pyplot as plt

    n_plots = 5
    n_pred = 50

    t0_col = 'time_after_upload_src'
    tf_col = 'time_after_upload_tgt'

    fig, axes = plt.subplots(n_plots, 3, sharex=True)

    data_ids_shuffle = data['nonbow_test'][KEYS_TRAIN_ID].drop_duplicates().sample(frac=1)
    for i, (_, row) in enumerate(data_ids_shuffle.iterrows()):
        if i < n_plots:
            # get raw_data for this video
            username = row[COL_USERNAME]
            video_id = row[COL_VIDEO_ID]

            df_i = data['nonbow_test']
            df_i = df_i[(df_i[COL_USERNAME] == username) * (df_i[COL_VIDEO_ID] == video_id)]
            df_i = df_i.sort_values(by=[t0_col])
            N = len(df_i)

            rec_last = df_i.iloc[df_i[t0_col].argmax()]
            idxs_start = [0, int(0.25 * N), int(0.5 * N), int(0.75 * N)]

            if 1: # extrapolate out from a few anchor measurements
                # create test sets anchored on a few src times
                df_test = []
                for j in idxs_start:
                    rec_first = df_i.iloc[j]
                    df_ = pd.DataFrame([rec_first for _ in range(n_pred)])
                    df_[tf_col] = np.linspace(rec_first[t0_col], rec_last[t0_col] * 1.25, n_pred)
                    df_test.append(df_)

                # predict with model
                df_pred = [model.predict(df_) for df_ in df_test]
            else: # incremental prediction from a starting time
                df_pred = incremental_pred(idxs_start, model, df_i, t0_col, rec_last, n_pred)

            # plot
            keys_tgt = KEYS_TRAIN_NUM_TGT
            for j, key in enumerate(keys_tgt):
                axes[i, j].plot(df_i[t0_col], df_i[key + '_src'])  # obs
                for df_ in df_pred:
                    axes[i, j].plot(df_[tf_col], df_[key + '_pred'])  # pred
            if i == 0:
                for j, key in enumerate(keys_tgt):
                    axes[i, j].set_title(key)
            axes[i, 0].set_ylabel(username + '\n' + video_id)

def incremental_pred(idxs_start, model, df_i, t0_col, rec_last, n_pred) -> List[pd.DataFrame]:
    """Incremental extrapolation of linear model predictions into the future from a starting time"""
    df_pred = []

    # create test sets anchored on a few src times
    for j in idxs_start:
        rec_first = df_i.iloc[j]
        df_ = pd.DataFrame([rec_first])
        times_ = np.linspace(rec_first[t0_col], rec_last[t0_col] * 1.25, n_pred)

        df_pred_ = []
        for k, t_ in enumerate(times_):
            if k == 0:
                # first sample
                df_pred_one = df_[KEYS_FOR_PRED_NONBOW_ID].copy()
                df_pred_one[KEY_TRAIN_TIME_DIFF + '_tgt'] = df_pred_one[KEY_TRAIN_TIME_DIFF + '_src']
                for key in KEYS_TRAIN_NUM_TGT:
                    df_pred_one[key + '_pred'] = df_[key + '_src']
            else:
                # predict one step forward (update src time, tgt time, and src stats)
                df_test = df_[KEYS_FOR_PRED_NONBOW_ID].copy()
                df_test[KEY_TRAIN_TIME_DIFF + '_src'] = df_pred_[-1][KEY_TRAIN_TIME_DIFF + '_tgt']
                df_test[KEY_TRAIN_TIME_DIFF + '_tgt'] = t_
                for key in KEYS_TRAIN_NUM_TGT:
                    df_test[key + '_src'] = df_pred_[-1][key + '_pred']
                for key in ['subscriber_count_src']:
                    df_test[key] = df_[key]
                df_pred_one = model.predict(df_test)
            df_pred_.append(df_pred_one)
        df_pred.append(pd.concat(df_pred_, axis=0, ignore_index=True))

    return df_pred

def print_train_data_stats(data_all: Dict[str, pd.DataFrame],
                           ml_request: MLRequest):
    """List out training dataset information."""
    print('\n\n=== ML data ===')

    print(f'Data object has keys: {list(data_all.keys())}')

    for key, df in data_all.items():
        print(f'\nDataFrame "{key}" has {len(df)} rows.')# and columns {list(df.columns)}.')
        if key == 'bow':
            print(f'Column {FEATURES_VECTOR_COL} has length {len(df.iloc[0]["vec"])}')
        if 'nonbow' in key:
            uname_id_pairs = df[[COL_USERNAME, COL_VIDEO_ID]].drop_duplicates()
            print(f'Dataset {key} has {len(uname_id_pairs)} videos across '
                  f'{len(uname_id_pairs[COL_USERNAME].unique())} users.')
        print('')
        print_df_full(df.iloc[:3])

    print('\n=== ML request object ===')

    pprint(ml_request.get_config())

    print('\n')

def show_seq2seq_predictions(data: Dict[str, pd.DataFrame],
                             df_preds: pd.DataFrame):
    """Plot sequence-to-sequence predictions."""
    import matplotlib.pyplot as plt

    usernames = list(df_preds[COL_USERNAME].unique())
    n_vid_ids = 5

    for uname in usernames:
        fig, axes = plt.figure(nrows=len(KEYS_TRAIN_NUM), ncols=n_vid_ids, figsize=(12, 8))

        video_ids = list(df_preds.query(f"{COL_USERNAME} == {uname}")[COL_VIDEO_ID].unique())

        for j in range(n_vid_ids):
            video_id_ = video_ids[j]
            df_test_ = data['nonbow_test'].query(f"{COL_VIDEO_ID} == {video_id_}")
            df_pred_ = df_preds.query(f"{COL_VIDEO_ID} == {video_id_}")

            for i, key in enumerate(KEYS_TRAIN_NUM):
                axes[i, j].plot(df_test_[COL_TIMESTAMP_ACCESSED], df_test_[key], c='k')
                axes[i, j].plot(df_pred_[COL_TIMESTAMP_ACCESSED], df_pred_[key], c='b')

    plt.show()















""" Model save and load """
def make_model_obj(model_,
                   _id: str,
                   uname_: Optional[str] = '') \
        -> dict:
    return {
        MODEL_MODEL_OBJ: model_.encode(),
        MODEL_META_ID: _id,
        MODEL_SPLIT_NAME: uname_
    }

def save_reg_model(model_reg: Union[MLModelBaseRegressionSimple, Dict[str, MLModelBaseRegressionSimple]],
                   ml_request: MLRequest,
                   preconfig: Dict[str, str]):
    """Save regression model(s) to the model store along with configs."""
    from db_engines.mongodb_engine import MongoDBEngine

    db_ = ml_request.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_MODELS_NOSQL_DATABASE']
    collection_meta = db_['db_info']['DB_MODELS_NOSQL_COLLECTIONS']['meta']
    collection_models = db_['db_info']['DB_MODELS_NOSQL_COLLECTIONS']['models']

    # check that model is the right format/type
    is_lrs_model = (isinstance(model_reg, MLModelBaseRegressionSimple) or
                    is_dict_of_instances(model_reg, MLModelBaseRegressionSimple))

    assert is_lrs_model

    # get _id from timestamp
    ts_str = get_ts_now_str('ms')
    _, _id = convert_ts_fmt_for_mongo_id(ts_str)

    ### configs and other metadata
    # prepare object
    obj = dict(_id=_id, meta={}, config={})
    if is_lrs_model:
        obj['meta'][ML_MODEL_TYPE] = ML_MODEL_TYPE_LIN_PROJ_RAND
    else:
        raise Exception('Model type not recognized.')
    obj['config']['preconfig'] = preconfig
    obj['config']['ml'] = ml_request.get_config()

    # validate obj
    assert set(obj) == {'_id', 'meta', 'config'}
    assert set([ML_MODEL_TYPE]) == set(obj['meta'])
    assert set(obj['config']) == {'preconfig', 'ml'}

    # write to model store
    engine = MongoDBEngine(mongo_config, database=database, collection=collection_meta, verbose=True)
    engine.insert_one(obj)

    ### models
    engine.set_db_info(collection=collection_models)

    # encode model(s) and write to store
    if isinstance(model_reg, dict): # multiple models
        for uname, model in model_reg.items():
            obj_ = make_model_obj(model, _id, uname)
            assert validate_mongodb_records_schema(obj_, SCHEMAS_MONGODB['models'])
            assert validate_mongodb_records_schema(obj_[MODEL_MODEL_OBJ], SCHEMAS_MONGODB['models_params'])
            engine.insert_one(obj_)
    else:
        obj_ = make_model_obj(model_reg, _id)
        assert validate_mongodb_records_schema(obj_, SCHEMAS_MONGODB['models'])
        assert validate_mongodb_records_schema(obj_[MODEL_MODEL_OBJ], SCHEMAS_MONGODB['models_params'])
        engine.insert_one(obj_)


