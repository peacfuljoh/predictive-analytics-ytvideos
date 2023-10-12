"""Machine learning models"""

from typing import List, Sequence, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.random_projection import SparseRandomProjection

from src.crawler.crawler.constants import FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS, \
    ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM, ML_HYPERPARAM_RLP_DENSITY
from src.crawler.crawler.utils.misc_utils import is_list_of_tuples
from src.ml.ml_request import MLRequest



class MLModel():
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
                assert is_list_of_tuples(pts, 2)
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


class MLModelLinProjRandom(MLModel):
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
