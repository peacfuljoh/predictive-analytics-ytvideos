"""Neural-network models"""

from typing import Union, Optional, Callable, Tuple, List, Iterator, Dict
import math
from collections import Counter

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
from src.ml.model_utils import _dfs_to_train_seqs


BATCH_SIZE = 5
SEQ_LEN_GROUP_WIDTH = 5 # width of divisions along seq len dimension

COL_SEQ_LEN_ORIG = 'seq_len_orig'
COL_SEQ_INFO_GROUP_ID = 'group_id'
COL_SEQ_LEN_GROUP = 'seq_len_group'


"""
Tensor of feature vector sequences:
- axis 0: sample index (0, ..., L-1)
- axis 1: sequence index (0, ..., N-1)
- axis 2: feature index (0, ..., D-1)
"""




class YTStatsDataset(Dataset):
    """
    YouTube statistics dataset.

    This Dataset class takes bow and non-bow DataFrames and prepares arrays for training for each video_id.
    Group information is also determined so that a sampler can group same-length sequences together.
    """
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

        # figure out grouping of sequences by length
        self._seq_info = pd.DataFrame(columns=[COL_SEQ_LEN_ORIG, COL_SEQ_INFO_GROUP_ID, COL_SEQ_LEN_GROUP])
        self._seq_info[COL_SEQ_LEN_ORIG] = data_nonbow.groupby(COL_VIDEO_ID).size() # access size via seq_lens[<video_id>]
        self._seq_info[COL_SEQ_INFO_GROUP_ID] = (self._seq_info[COL_SEQ_LEN_ORIG] / SEQ_LEN_GROUP_WIDTH).astype(int)
        self._seq_info[COL_SEQ_LEN_GROUP] = self._seq_info[COL_SEQ_INFO_GROUP_ID] * SEQ_LEN_GROUP_WIDTH

        # extract metadata
        # Note: data_bow acts like a metadata store for train and test combined, so it generally contains more
        #       video_id's than data_nonbow. That is why we can't get the relevant video_id's directly from data_bow.
        self._video_ids: List[str] = list(self._seq_info.index) # same order as index of seq_info
        self._num_seqs = len(self._video_ids)

        # store DataFrames
        # self._data_nonbow = data_nonbow.sort_values(by=[COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED])
        # self._data_bow = data_bow

        # prepare for quick access in __getitem__()
        self._data: Dict[str, Dict[str, np.ndarray]] = {}
        for video_id, df_nonbow in data_nonbow.groupby(COL_VIDEO_ID):
            # ensure time-sequential order
            df_nonbow = df_nonbow.sort_values(by=COL_TIMESTAMP_ACCESSED)

            # verify uniform sampling through time (COL_TIMESTAMP_ACCESSED entries are of type Timestamp())
            dt_seconds = df_nonbow[COL_TIMESTAMP_ACCESSED].apply(lambda x: x.timestamp()).to_numpy()
            dt_diffs = dt_seconds[1:] - dt_seconds[:-1]
            assert (np.abs(dt_diffs - dt_diffs[0]) < 1.0).all() # no more than 1 second error in sampling period

            # save sequences
            stats, embeds = _dfs_to_train_seqs(data_bow, df_nonbow)
            stats = stats[:self._seq_info[COL_SEQ_LEN_GROUP][video_id], :] # truncate to length for this seq's group
            self._data[video_id] = dict(stats=stats, embeds=embeds)

    def __len__(self):
        return self._num_seqs

    def __getitem__(self, idx: int):
        video_id: str = self._video_ids[idx]

        input_ = self._data[video_id]

        # input_ = input_[:, :4] # only stats, no embedded feats

        sample = {
            'stats': torch.tensor(input_['stats']),
            'embeds': torch.tensor(input_['embeds']),
            # 'target': torch.tensor(out_)
        }

        return sample



class YTStatsSampler(Sampler[List[int]]):
    """
    Batch sampler for variable-length sequence modeling.
    Generated batches are comprised of same-length sequences with uniform sampling over the entire dataset.
    """
    def __init__(self,
                 group_ids: List[int],
                 batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        # collect batch info for each group
        self._batch_info = {}
        g_ids_np = np.array(group_ids)
        for group_id in set(group_ids):
            idxs_ = list(np.where(g_ids_np == group_id)[0])
            num_batches = len(idxs_) // batch_size
            if num_batches == 0:
                continue
            self._batch_info[group_id] = {'idxs_in_dataset': idxs_, 'num_batches': num_batches}

        # [self._batch_info[group_id]['num_batches'] for group_id in self._batch_info.keys()] # batch counts in groups
        self._num_batches_total = sum([g['num_batches'] for g in self._batch_info.values()])

    def __len__(self) -> int:
        return self._num_batches_total

    def __iter__(self) -> Iterator[List[int]]:
        # make batches of data indices
        batches = []
        for group_id, d in self._batch_info.items():
            num_idxs_tot = self.batch_size * d['num_batches']
            idxs_perm = np.random.permutation(d['idxs_in_dataset'])[:num_idxs_tot]
            batches_group = idxs_perm.reshape(d['num_batches'], self.batch_size).tolist()
            batches += batches_group

        # scramble the batches
        batches = [batches[i] for i in np.random.permutation(len(batches))]

        # yield them
        for batch in batches:
            yield batch





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
        # dataset
        dataset = YTStatsDataset(data_nonbow, data_bow)

        # sampler
        group_ids = list(dataset._seq_info[COL_SEQ_LEN_GROUP])
        sampler = YTStatsSampler(group_ids, BATCH_SIZE)

        next(iter(sampler))
        return
        # dataloader
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