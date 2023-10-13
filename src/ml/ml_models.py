"""Machine learning models"""

from typing import List, Sequence, Union, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_SR_ALPHA,
                                           KEYS_TRAIN_ID, KEYS_TRAIN_NUM, KEYS_TRAIN_NUM_TGT, KEY_TRAIN_TIME_DIFF)
from src.crawler.crawler.utils.misc_utils import is_list_of_sequences
from src.ml.ml_request import MLRequest


KEYS_FOR_FIT_NONBOW_SRC = KEYS_TRAIN_NUM + [KEY_TRAIN_TIME_DIFF]
KEYS_FOR_FIT_NONBOW_SRC = [key + '_src' for key in KEYS_FOR_FIT_NONBOW_SRC]
KEYS_FOR_FIT_NONBOW_TGT = KEYS_TRAIN_NUM_TGT + [KEY_TRAIN_TIME_DIFF]
KEYS_FOR_FIT_NONBOW_TGT = [key + '_tgt' for key in KEYS_FOR_FIT_NONBOW_TGT]



class MLModelProjection():
    """Parent class for ML models."""
    def __init__(self, ml_request: MLRequest):
        # config
        assert ml_request.get_valid()
        self._config = ml_request.get_config()

        # fields
        self._n_features = None

    def _validate_df_bow(self, data):
        """Make sure data object is valid for bag-of-words models."""
        assert isinstance(data, pd.DataFrame)
        assert FEATURES_VECTOR_COL in data.columns
        if self._n_features is not None:
            for _, pts in self._bow_gen(data):
                assert is_list_of_sequences(pts, (list, tuple), 2)
                assert all([pt[0] < self._n_features for pt in pts])

    def _bow_gen(self, data: pd.DataFrame) -> List[tuple]:
        """Generator for sparse bag-of-words vectors in video feature DataFrame"""
        for i, pts in data[FEATURES_VECTOR_COL].items():
            yield i, pts
        return

    def _bow2csr(self, data: pd.DataFrame):
        """Pack bag-of-words list[tuple] into a sparse CSR array."""
        assert self._n_features is not None

        n_samples = len(data[FEATURES_VECTOR_COL])

        pts_all = dict(val=[], row=[], col=[])
        for i, pts in self._bow_gen(data):
            pts_all['val'] += [pt[1] for pt in pts]
            pts_all['col'] += [pt[0] for pt in pts]
            pts_all['row'] += [i] * len(pts)

        X = csr_array(
            (pts_all['val'], (pts_all['row'], pts_all['col'])),
            (n_samples, self._n_features)
        )

        return X


class MLModelLinProjRandom(MLModelProjection):
    """Random linear projection model"""
    def __init__(self, ml_request: MLRequest):
        super().__init__(ml_request)

        # ML request config
        assert self._config[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

        # model
        hp = self._config[ML_MODEL_HYPERPARAMS]
        self._model = SparseRandomProjection(
            hp[ML_HYPERPARAM_EMBED_DIM],
            density=hp[ML_HYPERPARAM_RLP_DENSITY],
            dense_output=True,
            random_state=None # int, for reproducibility
        )

    def fit(self, data: pd.DataFrame):
        """Fit model"""
        self._validate_df_bow(data)

        # determine data matrix dims
        n_samples = len(data[FEATURES_VECTOR_COL])
        self._n_features = 0
        for _, pts in self._bow_gen(data):
            self._n_features = max(self._n_features, max([pt[0] for pt in pts]))
        self._n_features += 1 # large index -> count

        # make sparse matrix for fit()
        X = csr_array((n_samples, self._n_features))#, dtype='float')

        # fit
        self._model.fit(X) # only uses shape

    def transform(self,
                  data: pd.DataFrame,
                  dtype = np.ndarray) \
            -> Union[pd.Series, np.ndarray]:
        """Apply random projection"""
        assert dtype in [pd.Series, np.ndarray]

        self._validate_df_bow(data)

        X = self._bow2csr(data)
        X_proj = self._model.transform(X)

        if dtype == np.ndarray:
            return X_proj
        if dtype == pd.Series:
            return pd.Series(X_proj.tolist())


class MLModelRegressionSimple():
    def __init__(self, ml_request: MLRequest):
        assert ml_request.get_valid()
        self._config = ml_request.get_config()

        # fields
        self._preprocessor: StandardScaler = StandardScaler()
        self._model: SGDRegressor = None

    def _fit_preprocessor(self,
                          data_nonbow: pd.DataFrame,
                          data_bow: pd.DataFrame):
        """Fit preprocessor before regression."""
        # TODO: partial_fit for incremental computation
        # fit preprocessor
        ss_bow = StandardScaler()
        X = np.array(list(data_bow[FEATURES_VECTOR_COL]))
        ss_bow.fit(X)

        ss_nonbow = StandardScaler()
        X = data_nonbow[KEYS_FOR_FIT_NONBOW_SRC].to_numpy()
        ss_nonbow.fit(X)

        self._preprocessor.scale_ = np.concatenate((ss_bow.scale_, ss_nonbow.scale_))
        self._preprocessor.mean_ = np.concatenate((ss_bow.mean_, ss_nonbow.mean_))
        self._preprocessor.var_ = np.concatenate((ss_bow.var_, ss_nonbow.var_))  # TODO: necessary?

    def _merge_dfs(self,
                   data_nonbow: pd.DataFrame,
                   data_bow: pd.DataFrame,
                   idxs_nonbow: Sequence[int] = None)  \
            -> np.ndarray:
        """Merge bow and non-bow dataframes into a single dense array."""
        # TODO: move out of class
        # TODO: test
        # select rows
        if idxs_nonbow is not None:
            data_nonbow = data_nonbow.iloc[idxs_nonbow]

        # find index mapping from bow df to nonbow df
        data_bow_midx = data_bow.set_index(KEYS_TRAIN_ID, drop=True)
        ids_ = data_nonbow[KEYS_TRAIN_ID].to_numpy().tolist()
        vecs = data_bow_midx.loc[ids_, FEATURES_VECTOR_COL]
        X_bow = np.array(list(vecs))

        # convert to numpy arrays
        X_nonbow = data_nonbow[KEYS_FOR_FIT_NONBOW_SRC].to_numpy()
        X = np.concatenate((X_bow, X_nonbow), axis=1)

        return X

    def fit(self,
            data_nonbow: pd.DataFrame,
            data_bow: pd.DataFrame):
        """Fit model to data."""
        # fit preprocessor
        self._fit_preprocessor(data_nonbow, data_bow)

        # fit regression model
        self._model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=self._config[ML_MODEL_HYPERPARAMS][ML_HYPERPARAM_SR_ALPHA],
            learning_rate='invscaling',
            eta0=0.01 # initial learning rate
        )



    def predict(self,
                data_nonbow: pd.DataFrame,
                data_bow: pd.DataFrame) \
            -> np.ndarray:
        """Predict"""
        pass
