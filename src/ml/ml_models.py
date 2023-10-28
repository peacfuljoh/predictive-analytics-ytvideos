"""Machine learning models"""

from typing import List, Union, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.random_projection import SparseRandomProjection
# from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS,
                                           ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM,
                                           ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_SR_ALPHAS, KEYS_TRAIN_ID,
                                           ML_HYPERPARAM_SR_CV_COUNT, ML_HYPERPARAM_SR_CV_SPLIT,
                                           KEYS_FOR_FIT_NONBOW_SRC, KEYS_FOR_FIT_NONBOW_TGT, KEYS_FOR_PRED_NONBOW_ID,
                                           KEYS_FOR_PRED_NONBOW_TGT, MODEL_DICT_PREPROCESSOR, MODEL_DICT_DATA_BOW,
                                           MODEL_DICT_MODEL, MODEL_DICT_CONFIG)
from ytpa_utils.val_utils import is_list_of_sequences
from ytpa_utils.df_utils import join_on_dfs, convert_mixed_df_to_array
from src.ml.ml_request import MLRequest




class MLModelProjection():
    """Parent class for ML models."""
    def __init__(self, ml_request: MLRequest):
        # config
        assert ml_request.get_valid()
        self._config = ml_request.get_config()

        # fields
        self._n_features = None

    def _validate_df_bow(self, data):
        """Make sure raw_data object is valid for bag-of-words models."""
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

        # determine raw_data matrix dims
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


class LinearRegressionCustom():
    def __init__(self, alpha: float = 0.0):
        self._alpha = alpha
        self._coeffs = None

    def _concat_const(self, X: np.ndarray) -> np.ndarray:
        """Concatenate constant value (ones) to feature matrix."""
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        """Fit model to raw_data."""
        _, N = X.shape
        Xs = self._concat_const(X)
        H = 1 / N * Xs.T @ Xs + self._alpha * np.eye(N + 1)
        V = 1 / N * Xs.T @ y
        self._coeffs = np.linalg.inv(H) @ V

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply model to make prediction for input raw_data."""
        return self._concat_const(X) @ self._coeffs

    def get_params_dict(self) -> dict:
        """Pack params into a dict for storage"""
        return dict(
            alpha=self._alpha,
            coeffs=self._coeffs.tolist()
        )

    def set_params_from_dict(self, d: dict):
        """Set parameters from storage dict format"""
        self._alpha = d['alpha']
        self._coeffs = np.array(d['coeffs'])


class MLModelRegressionSimple():
    def __init__(self,
                 ml_request: Optional[MLRequest] = None,
                 verbose: bool = False,
                 model_dict: Optional[dict] = None):
        self._verbose = verbose

        # fields
        self._data_bow: pd.DataFrame = None
        self._preprocessor: StandardScaler = StandardScaler()
        self._model: LinearRegressionCustom = LinearRegressionCustom()

        # process input args
        assert (ml_request is not None) ^ (model_dict is not None)

        if ml_request is not None:
            # validate and store config
            assert ml_request.get_valid()
            self._config = ml_request.get_config()
        else:
            # load model info from dict
            self.decode(model_dict)

    def _fit_preprocessor(self,
                          data_nonbow: pd.DataFrame):
        """Fit preprocessor before regression."""
        ss_bow = StandardScaler()
        X = np.array(list(self._data_bow[FEATURES_VECTOR_COL]))
        ss_bow.fit(X)

        ss_nonbow = StandardScaler()
        X = data_nonbow[KEYS_FOR_FIT_NONBOW_SRC].to_numpy()
        ss_nonbow.fit(X)

        # order matters - DO NOT change order
        self._preprocessor.scale_ = np.concatenate((ss_bow.scale_, ss_nonbow.scale_))
        self._preprocessor.mean_ = np.concatenate((ss_bow.mean_, ss_nonbow.mean_))
        self._preprocessor.var_ = np.concatenate((ss_bow.var_, ss_nonbow.var_))  # TODO: necessary?

    def _dfs_to_arrs(self,
                     data_nonbow: pd.DataFrame,
                     mode: str) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Join dataframes on IDs and construct input and output arrays."""
        assert mode in ['train', 'test']

        df_join = join_on_dfs(data_nonbow, self._data_bow, KEYS_TRAIN_ID, df1_keys_select=[FEATURES_VECTOR_COL])
        X = convert_mixed_df_to_array(df_join, [FEATURES_VECTOR_COL] + KEYS_FOR_FIT_NONBOW_SRC) # order matters - DO NOT change order
        y = None if mode == 'test' else convert_mixed_df_to_array(df_join, KEYS_FOR_FIT_NONBOW_TGT)

        return X, y

    def fit(self,
            data_nonbow: pd.DataFrame,
            data_bow: pd.DataFrame):
        """Fit model to raw_data."""
        hp = self._config[ML_MODEL_HYPERPARAMS]

        # validate and store bow raw_data
        # assert len(data_bow.drop_duplicates()) == len(data_bow) # no duplicated rows --> drop_duplicates() fails
        self._data_bow = data_bow

        # fit preprocessor
        self._fit_preprocessor(data_nonbow) # TODO: include preprocessor fit in cv splits

        # fit regression model
        X, y = self._dfs_to_arrs(data_nonbow, 'train')
        X = self._preprocessor.transform(X)
        N, _ = X.shape

        num_samps_all = X.shape[0]
        num_samps_train = int(num_samps_all * hp[ML_HYPERPARAM_SR_CV_SPLIT])

        num_splits = hp[ML_HYPERPARAM_SR_CV_COUNT]
        alphas = hp[ML_HYPERPARAM_SR_ALPHAS]

        models = [] # alphas x trials
        objs = []
        for i, alpha in enumerate(alphas):
            objs_ = []
            models_ = []

            for j in range(num_splits):
                # train-val split
                idxs = np.random.permutation(num_samps_all)
                idxs_train, idxs_val = idxs[:num_samps_train], idxs[num_samps_train:]
                X_train = X[idxs_train, :]
                y_train = y[idxs_train, :]
                X_val = X[idxs_val, :]
                y_val = y[idxs_val, :]

                # fit model
                model_ = LinearRegressionCustom(alpha=alpha)
                model_.fit(X_train, y_train)

                # eval on val set
                y_pred = model_.predict(X_val)
                obj_ = 1 / N * np.sum((y_val - y_pred) ** 2)

                if self._verbose:
                    print(f'LinearRegressionCustom: alpha={alpha}, split={j}, obj={obj_}.')

                # save results
                models_.append(model_)
                objs_.append(obj_)

            models.append(models_)
            objs.append(objs_)

        # pick best alpha value
        objs_avg = [np.mean(v) for v in objs]
        idx_best = np.argmin(objs_avg) # idx of winning alpha value
        alpha_best = alphas[idx_best]
        if self._verbose:
            d = {alpha_: obj_ for alpha_, obj_ in zip(alphas, objs_avg)}
            print(f'LinearRegressionCustom: cv_objs={d}.')

        # set model
        self._model = LinearRegressionCustom(alpha=alpha_best)
        self._model._coeffs = np.mean(np.stack([model_._coeffs for model_ in models[idx_best]], 2), axis=2)

    def predict(self,
                data_nonbow: pd.DataFrame,
                mode: Union[np.ndarray, pd.DataFrame] = pd.DataFrame) \
            -> Union[np.ndarray, pd.DataFrame]:
        """Predict"""
        assert mode in [np.ndarray, pd.DataFrame]

        X, _ = self._dfs_to_arrs(data_nonbow, 'test')
        X = self._preprocessor.transform(X)
        y_pred = self._model.predict(X)
        return self._convert_pred_to_df(data_nonbow, y_pred)

    def _convert_pred_to_df(self,
                            data_nonbow: pd.DataFrame,
                            y_pred: np.ndarray) \
        -> pd.DataFrame:
        """Convert prediction array into DataFrame to retain identifying details."""
        data_pred = data_nonbow[KEYS_FOR_PRED_NONBOW_ID].copy()
        for i, key in enumerate(KEYS_FOR_PRED_NONBOW_TGT):
            data_pred[key] = y_pred[:, i]
        return data_pred

    def encode(self) -> dict:
        """Convert model info to storage dict."""
        d = {}
        d[MODEL_DICT_CONFIG] = self._config
        d[MODEL_DICT_DATA_BOW] = self._data_bow.to_dict('records') # drops record indices
        d[MODEL_DICT_PREPROCESSOR] = dict(
            name='StandardScaler',
            params=dict(
                scale_ = list(self._preprocessor.scale_),
                mean_ = list(self._preprocessor.mean_),
                var_ = list(self._preprocessor.var_)
            )
        )
        d[MODEL_DICT_MODEL] = dict(
            name='LinearRegressionCustom',
            params=self._model.get_params_dict()
        )

        return d

    def decode(self, model_dict: dict):
        """Load model from storage dict."""
        self._config = model_dict[MODEL_DICT_CONFIG]
        self._data_bow = pd.DataFrame.from_dict(model_dict[MODEL_DICT_DATA_BOW])

        pre = model_dict[MODEL_DICT_PREPROCESSOR]
        assert pre['name'] == 'StandardScaler'
        par = pre['params']
        self._preprocessor.scale_ = np.array(par['scale_'])
        self._preprocessor.mean_ = np.array(par['mean_'])
        self._preprocessor.var_ = np.array(par['var_'])

        mod = model_dict[MODEL_DICT_MODEL]
        assert mod['name'] == 'LinearRegressionCustom'
        self._model.set_params_from_dict(mod['params'])













# self._model = SGDRegressor(
        #     loss='squared_error',
        #     penalty='l2',
        #     alpha=self._config[ML_MODEL_HYPERPARAMS][ML_HYPERPARAM_SR_ALPHA],
        #     learning_rate='invscaling',
        #     eta0=0.01 # initial learning rate
        # )