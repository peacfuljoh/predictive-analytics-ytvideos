"""Utils for model fit and predict internals"""
import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from ytpa_utils.df_utils import join_on_dfs, convert_mixed_df_to_array
from ytpa_utils.io_utils import load_pickle

from src.crawler.crawler.config import MODEL_ROOT
from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, KEYS_FOR_FIT_NONBOW_SRC,
                                           KEYS_FOR_FIT_NONBOW_TGT, KEYS_TRAIN_NUM, COL_VIDEO_ID, MODEL_ID)
from src.ml.ml_request import MLRequest


# from src.rnnoise.constants import DEVICE, MODELS_DIR, MODEL_TYPES


def _dfs_to_arrs_with_src_tgt(data_bow: pd.DataFrame,
                              data_nonbow: pd.DataFrame,
                              mode: str) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Join dataframes on IDs and construct input and output arrays. Some stats are given _src and _tgt suffixes."""
    assert mode in ['train', 'test']

    df_join = join_on_dfs(data_nonbow, data_bow, KEYS_TRAIN_ID, df1_keys_select=[FEATURES_VECTOR_COL])
    X = convert_mixed_df_to_array(df_join, [FEATURES_VECTOR_COL] + KEYS_FOR_FIT_NONBOW_SRC)  # order matters - DO NOT change order
    y = None if mode == 'test' else convert_mixed_df_to_array(df_join, KEYS_FOR_FIT_NONBOW_TGT)

    return X, y

def _dfs_to_arr_seq(data_bow: pd.DataFrame,
                    data_nonbow: pd.DataFrame) \
        -> np.ndarray:
    """Join dataframes on IDs and construct input and output arrays."""
    df_join = join_on_dfs(data_nonbow, data_bow, KEYS_TRAIN_ID, df1_keys_select=[FEATURES_VECTOR_COL])
    X = convert_mixed_df_to_array(df_join, KEYS_TRAIN_NUM + [FEATURES_VECTOR_COL])

    return X

def _dfs_to_train_seqs(data_bow: pd.DataFrame,
                       data_nonbow: pd.DataFrame) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Get time-independent and -dependent data in arrays. Rows of data_nonbow are assumed to be pre-sorted."""
    video_id = data_nonbow.iloc[0][COL_VIDEO_ID]
    assert (data_nonbow[COL_VIDEO_ID] == video_id).all() # ensure that only one video is being processed

    arr_td = data_nonbow[KEYS_TRAIN_NUM].to_numpy() # time-dependent 2D array
    df_ti = data_bow[data_bow[COL_VIDEO_ID] == video_id].iloc[0][FEATURES_VECTOR_COL]
    arr_ti = np.array(list(df_ti)) # time-independent 1D array

    return arr_td, arr_ti


def load_models(ml_request: MLRequest) -> Dict[str, dict]:
    """Load trained model info"""
    # get model id and corresponding subdir
    config_ml = ml_request.get_config()
    model_id = config_ml[MODEL_ID]
    model_subdir = os.path.join(MODEL_ROOT, model_id)

    # get pickle filenames in the subdir for this run
    fnames = [fname for fname in sorted(os.listdir(model_subdir)) if '.pickle' in fname]
    dt_strs = [fname.split('__')[0] for fname in fnames] # get sub-model datetimes # TODO: do this with a regex
    dt_strs = list(np.unique(dt_strs))

    # get latest models
    models_info = {}
    for model_sub_id in dt_strs:
        # get fnames for this model and find latest snapshot
        fnames_snapshot = [fname for fname in fnames if model_sub_id in fname]
        idx_ = np.argmax([int(fname.split('__')[-1].split('.')[0]) for fname in fnames_snapshot])
        fname_to_load = fnames_snapshot[idx_]

        # load model info
        fpath_to_load = os.path.join(model_subdir, fname_to_load)
        models_info[model_sub_id] = load_pickle(fpath_to_load)

    return models_info
