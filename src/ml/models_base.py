"""Base classes for ML models"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from src.ml.ml_request import MLRequest


class MLModelBase():
    def __init__(self,
                 ml_request: Optional[MLRequest] = None,
                 verbose: bool = False,
                 model_dict: Optional[dict] = None):
        self._verbose = verbose

        # fields
        self._data_bow: pd.DataFrame = None
        self._config = None

        # validate input args
        assert (ml_request is not None) ^ (model_dict is not None)

        if ml_request is not None:
            # validate and store config
            assert ml_request.get_valid()
            self._config = ml_request.get_config()
        else:
            # load model info from dict
            self.decode(model_dict)

    def fit(self,
            data_nonbow: pd.DataFrame, # [KEYS_FOR_FIT_NONBOW_SRC]
            data_bow: pd.DataFrame): # [username, video_id, vec]):
        raise NotImplementedError

    def predict(self,
                data_nonbow: pd.DataFrame,
                mode: Union[np.ndarray, pd.DataFrame] = pd.DataFrame) \
            -> Union[np.ndarray, pd.DataFrame]:
        raise NotImplementedError

    def encode(self) -> dict:
        """Convert model info to storage dict."""
        raise NotImplementedError

    def decode(self, model_dict: dict):
        raise NotImplementedError
