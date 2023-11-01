"""Train utils"""

from typing import Generator, Tuple, Dict, List, Optional, Union
import copy

import pandas as pd
import numpy as np

from db_engines.mongodb_utils import get_mongodb_records_gen
from db_engines.mongodb_engine import MongoDBEngine
from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL,
                                           TIMESTAMP_FMT, MIN_VID_SAMPS_FOR_DATASET,
                                           NUM_INTVLS_PER_VIDEO, ML_MODEL_TYPE, ML_MODEL_TYPE_LIN_PROJ_RAND, TRAIN_TEST_SPLIT,
                                           KEYS_TRAIN_ID, KEYS_TRAIN_NUM, KEYS_TRAIN_NUM_TGT,
                                           KEY_TRAIN_TIME_DIFF, SPLIT_TRAIN_BY_USERNAME, KEYS_FOR_PRED_NONBOW_ID,
                                           COL_VIDEO_ID, COL_USERNAME, COL_TIMESTAMP_ACCESSED,
                                           MODEL_MODEL_OBJ, MODEL_META_ID, MODEL_SPLIT_NAME)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import (load_config_timestamp_sets_for_features,
                                                              convert_ts_fmt_for_mongo_id)
from ytpa_utils.val_utils import is_dict_of_instances
from ytpa_utils.time_utils import get_ts_now_str
from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom, MLModelRegressionSimple
from src.schemas.schema_validation import validate_mongodb_records_schema
from src.schemas.schemas import SCHEMAS_MONGODB


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']

DB_MODELS_NOSQL_DATABASE = DB_INFO['DB_MODELS_NOSQL_DATABASE']
DB_MODELS_NOSQL_COLLECTIONS = DB_INFO['DB_MODELS_NOSQL_COLLECTIONS']



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

    for ids, df in df_data.groupby(KEYS_TRAIN_ID): # one group per video
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

def prepare_feature_records(df_gen: Generator[pd.DataFrame, None, None],
                            ml_request: MLRequest) \
        -> Tuple[Dict[str, pd.DataFrame], MLModelLinProjRandom]:
    """Stream raw_data in and convert to format needed for ML"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

    # setup
    keys_extract = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL, COL_TIMESTAMP_ACCESSED] + KEYS_TRAIN_NUM  # define cols to keep
    keys_feat_src = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]  # columns of interest for output vectors
    keys_feat_tgt = KEYS_TRAIN_NUM_TGT + [KEY_TRAIN_TIME_DIFF] # columns of interest for output vectors

    # stream all raw_data into RAM
    df_data = stream_all_features_into_ram(df_gen, keys_extract)

    # filter by group and collect bag-of-words info in a separate DataFrame
    df_nonbow, df_bow = split_feature_df_and_filter(df_data, MIN_VID_SAMPS_FOR_DATASET)

    # embed bag-of-words features: raw_data-independent dimensionality reduction
    model_embed = embed_bow_with_lin_proj_rand(ml_request, df_bow) # df_bow modified in-place

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
                                  ml_request: MLRequest) \
        -> Union[MLModelRegressionSimple, Dict[str, MLModelRegressionSimple]]:
    """Train simple regression model"""
    config_ml = ml_request.get_config()
    assert config_ml[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

    # fit model
    if not config_ml[SPLIT_TRAIN_BY_USERNAME]:
        model = MLModelRegressionSimple(ml_request, verbose=True)
        model.fit(data['nonbow_train'], data['bow'])
    else:
        usernames = data['nonbow_train'][COL_USERNAME].drop_duplicates()
        model = {uname: MLModelRegressionSimple(ml_request, verbose=True) for uname in usernames}
        for uname, model_ in model.items():
            df_nonbow = data['nonbow_train']
            df_nonbow = df_nonbow[df_nonbow[COL_USERNAME] == uname]
            df_bow = data['bow']
            df_bow = df_bow[df_bow[COL_USERNAME] == uname]
            model_.fit(df_nonbow, df_bow)

        if 1:
            # try model encoding and decoding
            model_dicts = {}
            for uname, model_ in model.items():
                model_dicts[uname] = model_.encode()
            model = {}
            for uname in usernames:
                model[uname] = MLModelRegressionSimple(verbose=True, model_dict=model_dicts[uname])

    # see predictions
    if 0:
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
                # TODO: move the incremental prediction block to its own function (to within lin reg class?)
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

def make_model_obj(model_,
                   _id: str,
                   uname_: Optional[str] = '') \
        -> dict:
    return {
        MODEL_MODEL_OBJ: model_.encode(),
        MODEL_META_ID: _id,
        MODEL_SPLIT_NAME: uname_
    }

def save_reg_model(model_reg: Union[MLModelRegressionSimple, Dict[str, MLModelRegressionSimple]],
                   ml_request: MLRequest,
                   config_load: Dict[str, str]):
    """Save regression model(s) to the model store along with configs."""
    # check that model is the right format/type
    is_lrs_model = (isinstance(model_reg, MLModelRegressionSimple) or
                    is_dict_of_instances(model_reg, MLModelRegressionSimple))

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
    obj['config']['load'] = config_load
    obj['config']['ml'] = ml_request.get_config()

    # validate obj
    assert set(obj) == {'_id', 'meta', 'config'}
    assert set([ML_MODEL_TYPE]) == set(obj['meta'])
    assert set(obj['config']) == {'load', 'ml'}

    # write to model store
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_MODELS_NOSQL_DATABASE,
                           collection=DB_MODELS_NOSQL_COLLECTIONS['meta'],
                           verbose=True)
    engine.insert_one(obj)

    ### models
    engine.set_db_info(collection=DB_MODELS_NOSQL_COLLECTIONS['models'])

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


