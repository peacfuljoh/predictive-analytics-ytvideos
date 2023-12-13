"""Neural-network models"""

from typing import Union, Optional, Callable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.ml.ml_request import MLRequest
from src.ml.ml_constants import KEYS_EXTRACT_SEQ2SEQ
from src.crawler.crawler.constants import KEYS_TRAIN_ID, FEATURES_VECTOR_COL, COL_VIDEO_ID
from src.ml.model_utils import _dfs_to_arr_seq


BATCH_SIZE = 5


"""
Tensor of feature vector sequences:
- axis 0: sample index (0, ..., L-1)
- axis 1: sequence index (0, ..., N-1)
- axis 2: feature index (0, ..., D-1)
"""


class YTStatsDataset(Dataset):
    """YouTube statistics dataset."""
    def __init__(self,
                 data_nonbow: pd.DataFrame,
                 data_bow: pd.DataFrame,
                 transform: Optional[Callable] = None):
        # verify columns
        keys_nonbow = [key for key in KEYS_EXTRACT_SEQ2SEQ if key is not FEATURES_VECTOR_COL]
        keys_bow = [KEYS_TRAIN_ID + [FEATURES_VECTOR_COL]]
        assert set(data_nonbow) == set(keys_nonbow)
        assert set(data_bow) == set(keys_bow)

        # extract metadata
        self._video_ids = data_bow[COL_VIDEO_ID].unique()
        self._num_samps = len(self._video_ids)
        assert self._num_samps == len(data_bow)

        self._data_nonbow = data_nonbow
        self._data_bow = data_bow
        self.transform = transform

    def __len__(self):
        return self._num_samps

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_id_: str = self._video_ids[idx]
        mask = self._data_nonbow[COL_VIDEO_ID] == video_id_
        arr: np.ndarray = _dfs_to_arr_seq(self._data_bow, self._data_nonbow[mask])
        sample = {'seq': torch.tensor(arr)} # TODO: ensure correct shape (e.g. Lx1xD)

        if self.transform:
            sample = self.transform(sample)

        return sample



class MLModelSeq2Seq():
    def __init__(self,
                 ml_request: MLRequest,
                 verbose: bool = False):
        self._verbose = verbose

        # fields
        self._config = None
        self._model = None

        # validate and store config
        assert ml_request.get_valid()
        self._config = ml_request.get_config()

    def fit(self,
            data_nonbow: pd.DataFrame,
            data_bow: pd.DataFrame):
        dataset = YTStatsDataset(data_nonbow, data_bow)

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)

        # TODO: implement this
        pass

    def predict(self,
                data_nonbow: pd.DataFrame,
                mode: Union[np.ndarray, pd.DataFrame] = pd.DataFrame) \
            -> Union[np.ndarray, pd.DataFrame]:
        raise NotImplementedError


