"""Neural-network models"""

from typing import Union, Optional, Callable, Tuple, List, Iterator

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler

from src.ml.ml_request import MLRequest
from src.ml.ml_constants import KEYS_EXTRACT_SEQ2SEQ
from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED,
                                           COL_USERNAME)
from src.ml.model_utils import _dfs_to_arr_seq


BATCH_SIZE = 5
SEQ_LEN_GROUP_WIDTH = 5 # width of divisions along seq len dimension


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
                 data_bow: pd.DataFrame):
        """
        Initialize the dataset.

        Args:
            data_nonbow: DataFrame where each row has a timestamped sample of video stats
            data_bow: DataFrame with one row per video containing embedded static features (e.g. text, image)
        """
        # verify columns
        keys_nonbow = [key for key in KEYS_EXTRACT_SEQ2SEQ if key is not FEATURES_VECTOR_COL]
        keys_bow = KEYS_TRAIN_ID + [FEATURES_VECTOR_COL]
        assert set(data_nonbow) == set(keys_nonbow)
        assert set(data_bow) == set(keys_bow)

        # extract metadata
        # Note: data_bow acts like a metadata store for train and test combined, so it generally contains more
        #       video_id's than data_nonbow. That is why we can't get the relevant video_id's directly data_bow.
        self._video_ids: List[str] = list(data_nonbow[COL_VIDEO_ID].unique())
        self._num_seqs = len(self._video_ids)

        # figure out grouping of sequences by length
        seq_lens = data_nonbow.groupby(COL_VIDEO_ID).size() # access size via self._seq_lens[<video_id_string>]
        self._seq_info = pd.DataFrame({'seq_len': seq_lens, 'group': (seq_lens / SEQ_LEN_GROUP_WIDTH).astype(int)})
        self._seq_info['seq_len_group'] = self._seq_info['group'] * SEQ_LEN_GROUP_WIDTH

        # store DataFrames
        self._data_nonbow = data_nonbow.sort_values(by=[COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED])
        self._data_bow = data_bow

    def __len__(self):
        return self._num_seqs

    def __getitem__(self, idx: int):
        video_id_: str = self._video_ids[idx]

        mask = self._data_nonbow[COL_VIDEO_ID] == video_id_
        input_: np.ndarray = _dfs_to_arr_seq(self._data_bow, self._data_nonbow[mask])

        # input_ = input_[:, :4] # only stats, no embedded feats

        sample = {'input': torch.tensor(input_)}#, 'target': torch.tensor(out_)}

        return sample



class SequenceBatchSampler(Sampler[List[int]]):
    def __init__(self,
                 data: List[str],
                 batch_size: int):
        super().__init__()

        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        sizes = torch.tensor([len(x) for x in self.data])
        for batch in torch.chunk(torch.argsort(sizes), len(self)):
            yield batch.tolist()





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
        # setup
        dataset, dataloader = self._prepare_dataset_and_dataloader(data_nonbow, data_bow)
        self._prepare_train_modules()

        model = Seq2Seq()

        for i in range(len(dataset)):
            print(f'Iter: {i}')
            x = dataset[i]
            a = 5

        # run training epochs
        pass

    @staticmethod
    def _prepare_dataset_and_dataloader(data_nonbow: pd.DataFrame,
                                        data_bow: pd.DataFrame) \
            -> Tuple[YTStatsDataset, DataLoader]:
        """Make dataset and dataloader for a training run"""
        dataset = YTStatsDataset(data_nonbow, data_bow)
        sampler = SequenceBatchSampler()
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                shuffle=True, num_workers=0, pin_memory=False)

        if 1:
            # see histogram of sequence lengths
            import matplotlib.pyplot as plt

            bin_width = 5 # measurements (i.e. hours)
            plt.hist(dataset._seq_info['seq_len'], bins=np.arange(0, 7 * 24 + bin_width, bin_width))
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