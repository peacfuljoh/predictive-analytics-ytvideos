"""Machine learning op config"""

from typing import List

import copy
import math

import pandas as pd
import numpy as np
from scipy.sparse import csr_array

from sklearn.random_projection import SparseRandomProjection

from src.crawler.crawler.constants import FEATURES_VECTOR_COL


ML_MODEL_TYPE = 'model_type'
ML_MODEL_HYPERPARAMS = 'hyperparams'
ML_MODEL_TYPE_LIN_PROJ_RAND = 'lin_proj_random'
ML_MODEL_TYPE_GAM_TOPIC = 'gam_topic'

ML_HYPERPARAM_EMBED_DIM = 'embed_dim'
ML_HYPERPARAM_RLP_DENSITY = 'density'

ML_CONFIG_KEYS = [ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS]
ML_MODEL_TYPES = [ML_MODEL_TYPE_LIN_PROJ_RAND, ML_MODEL_TYPE_GAM_TOPIC]


class MLRequest():
    """ML process request object. Facilitates config validation and handling."""
    def __init__(self,
                 config: dict):
        self._valid = False

        self._config = self._validate_config(config)

    def _validate_config(self, config: dict) -> dict:
        """Validate all config options"""
        config = copy.deepcopy(config) # in case there are in-place modifications

        # make sure all mandatory keys are present
        for key in ML_CONFIG_KEYS:
            assert key in config

        # validate model type
        assert config[ML_MODEL_TYPE] in ML_MODEL_TYPES

        # validate by specific model types
        if config[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND:
            self._validate_lin_proj_rand(config)

        # mark request as valid
        self._valid = True

        return config

    def _validate_lin_proj_rand(self, config: dict):
        """Validate config options specific to the random linear projection model."""
        # validate hyperparameters
        hp = config[ML_MODEL_HYPERPARAMS]

        assert ML_HYPERPARAM_EMBED_DIM in hp

        hp[ML_HYPERPARAM_RLP_DENSITY] = hp.get(ML_HYPERPARAM_RLP_DENSITY, 'auto') # if 'auto': 1 / sqrt(n_features)
        if not isinstance(hp[ML_HYPERPARAM_RLP_DENSITY], str): # must be float
            assert 0.0 < hp[ML_HYPERPARAM_RLP_DENSITY] < 1.0

    def get_config(self) -> dict:
        assert self.get_valid()
        return copy.deepcopy(self._config)

    def get_valid(self) -> bool:
        return self._valid



class MLModelLinProjRandom():
    """Random linear projection model"""
    def __init__(self, ml_request: MLRequest):
        self._config = ml_request.get_config()

        self._n_features = None

        assert self._config[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND

        hp = self._config[ML_MODEL_HYPERPARAMS]

        self._model = SparseRandomProjection(
            hp[ML_HYPERPARAM_EMBED_DIM],
            density=hp[ML_HYPERPARAM_RLP_DENSITY],
            dense_output=True,
            random_state=None # int, for reproducibility
        )

    def _validate_df(self, data):
        assert isinstance(data, pd.DataFrame)
        assert FEATURES_VECTOR_COL in data.columns
        if self._n_features is not None:
            for _, pts in self._bow_gen(data):
                assert max([pt[0] for pt in pts]) < self._n_features

    def _bow_gen(self, data: pd.DataFrame) -> List[tuple]:
        """Generator for sparse bag-of-words vectors in video feature DataFrame"""
        for i, pts in data[FEATURES_VECTOR_COL].items():
            yield i, pts
        yield StopIteration

    def fit(self, data: pd.DataFrame):
        """Fit model"""
        self._validate_df(data)

        # determine data matrix dims
        n_samples = len(data[FEATURES_VECTOR_COL])
        self._n_features = 0
        for _, pts in self._bow_gen(data):
            self._n_features = max(self._n_features, max([pt[0] for pt in pts]))
        self._n_features += 1

        # make sparse matrix for fit()
        X = csr_array((n_samples, self._n_features))#, dtype='float')

        # fit (doesn't use data values, only shape)
        self._model.fit(X)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Apply random projection"""
        self._validate_df(data)

        # convert to csr matrix
        n_samples = len(data[FEATURES_VECTOR_COL])
        pts_all = dict(val=[], row=[], col=[])
        for i, pts in self._bow_gen(data):
            for pt in pts:
                pts_all['val'].append(pt[1])
                pts_all['row'].append(i)
                pts_all['col'].append(pt[0])
        X = csr_array((pts_all['val'], (pts_all['row'], pts_all['col'])), (n_samples, self._n_features))

        # project
        X_proj = self._model.transform(X)

        return X_proj

