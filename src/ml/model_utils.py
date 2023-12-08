"""Utils for model fit and predict internals"""

from typing import Tuple

import numpy as np
import pandas as pd
from ytpa_utils.df_utils import join_on_dfs, convert_mixed_df_to_array

from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, KEYS_FOR_FIT_NONBOW_SRC,
                                           KEYS_FOR_FIT_NONBOW_TGT, KEYS_TRAIN_NUM)


def _dfs_to_arrs_with_src_tgt(data_bow: pd.DataFrame,
                              data_nonbow: pd.DataFrame,
                              mode: str) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Join dataframes on IDs and construct input and output arrays."""
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
    X = convert_mixed_df_to_array(df_join, [FEATURES_VECTOR_COL, KEYS_TRAIN_NUM])

    return X

