"""Neural-network models"""

from typing import Union, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ytpa_utils.val_utils import is_subset

from src.ml.ml_request import MLRequest
from src.crawler.crawler.constants import COL_VIDEO_ID
from src.ml.model_utils import YTStatsDataset, YTStatsSampler
from src.ml.ml_constants import SEQ_LEN_GROUP_WIDTH, COL_SEQ_LEN_ORIG, COL_SEQ_LEN_GROUP, TRAIN_BATCH_SIZE



"""
Tensor of feature vector sequences:
- axis 0: sample index (0, ..., L-1)
- axis 1: sequence index (0, ..., N-1)
- axis 2: feature index (0, ..., D-1)
"""



class MLModelSeq2Seq():
    """Wrapper class for handling sequence-to-sequence model train and test"""
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
        """Fit the model with static (bow) and dynamic (nonbow) features."""
        # make sure metadata is available for all videos
        assert is_subset(list(data_nonbow[COL_VIDEO_ID]), list(data_bow[COL_VIDEO_ID]))

        # setup
        dataloader = self._prepare_dataset_and_dataloader(data_nonbow, data_bow)
        self._prepare_train_modules()

        model = Seq2Seq()

        for i in range(len(dataloader.dataset)):
            print(f'Iter: {i}')
            x = dataloader.dataset[i]
            a = 5

        # run training epochs
        pass

    @staticmethod
    def _prepare_dataset_and_dataloader(data_nonbow: pd.DataFrame,
                                        data_bow: pd.DataFrame) \
            -> Tuple[YTStatsDataset, DataLoader]:
        """Make dataset and dataloader for a training run"""
        # dataset
        dataset = YTStatsDataset(data_nonbow, data_bow)

        # sampler
        group_ids = list(dataset._seq_info[COL_SEQ_LEN_GROUP])
        sampler = YTStatsSampler(group_ids, TRAIN_BATCH_SIZE)

        # dataloader
        dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=False)

        if 0:
            # see histogram of sequence lengths
            import matplotlib.pyplot as plt

            bin_width = SEQ_LEN_GROUP_WIDTH  # measurements (i.e. hours)
            data_ = dataset._seq_info[COL_SEQ_LEN_ORIG]
            max_val = data_.max()

            fig, ax = plt.subplots(1, 1)
            ax.hist(data_, bins=np.arange(0, max_val + bin_width, bin_width))
            ax.set_ylabel('Count')
            ax.set_xlabel('Sequence length')

            plt.show()

        return dataset, dataloader

    @staticmethod
    def _prepare_train_modules():
        """Set up training modules, hyperparameters, etc."""
        pass

    def predict(self,
                data_nonbow: pd.DataFrame,
                mode: Union[np.ndarray, pd.DataFrame] = pd.DataFrame) \
            -> Union[np.ndarray, pd.DataFrame]:
        raise NotImplementedError


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence recurrent model.

    Ref: https://arxiv.org/abs/1409.3215
    """
    def __init__(self):
        super().__init__()

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass