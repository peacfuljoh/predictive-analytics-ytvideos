"""Dataset and related classes"""

import math
from typing import List, Dict, Tuple, Iterator, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, KEYS_TRAIN_ID, COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED,
                                           KEYS_TRAIN_NUM_STATIC_IDXS, KEYS_TRAIN_NUM_TGT_IDXS)
from src.ml.ml_constants import KEYS_EXTRACT_SEQ2SEQ, COL_SEQ_LEN_ORIG, COL_SEQ_INFO_GROUP_ID, COL_SEQ_LEN_GROUP, \
    SEQ_LEN_GROUP_WIDTH
from src.ml.model_utils import _dfs_to_train_seqs




class YTStatsDataset(Dataset):
    """
    YouTube statistics dataset.

    This Dataset class takes bow and non-bow DataFrames and prepares arrays for training for each video_id.
    Group information is also determined so that a sampler can group same-length sequences together.
    """
    def __init__(self,
                 data_nonbow: pd.DataFrame,
                 data_bow: pd.DataFrame,
                 mode: str):
        """
        Initialize the dataset.

        Args:
            data_nonbow: DataFrame where each row has a timestamped sample of video stats
            data_bow: DataFrame with one row per video containing embedded static features (e.g. text, image)
        """
        assert mode in ['train', 'test', 'predict']

        self._mode = mode

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
        self._data: Dict[str, Dict[str, torch.Tensor]] = {}
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
            self._data[video_id] = dict(
                stats=torch.tensor(stats, dtype=torch.float),
                embeds=torch.tensor(embeds, dtype=torch.float)
            )

    def __len__(self):
        return self._num_seqs

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Union[int, torch.Tensor]]:
        video_id: str = self._video_ids[idx]
        data_ = self._data[video_id]

        # embeddings and other static (non-time-varying) features
        embeds_ = torch.concat((
            data_['embeds'],
            data_['stats'][0, KEYS_TRAIN_NUM_STATIC_IDXS]
        )) # num_embeds_features + 1

        # dynamic (time-varying) features
        stats_ = data_['stats'][:-1, KEYS_TRAIN_NUM_TGT_IDXS] # (num_steps - 1) x (num_stats_features - 1)

        # pack tensors and other info into input and output objects
        input_ = {
            'video_id': video_id,
            'stats': stats_,
            'embeds': embeds_
        }

        output_ = 0
        if self._mode in ['train', 'test']:
            output_ = data_['stats'][1:, KEYS_TRAIN_NUM_TGT_IDXS] # (num_steps - 1) x (num_stats_features - 1)
        elif self._mode == 'predict':
            pass
        else:
            raise NotImplementedError

        return input_, output_


class VariableLengthSequenceBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for variable-length sequence modeling.
    Generated batches are comprised of same-length sequences with uniform sampling over the entire dataset.
    """
    def __init__(self,
                 group_ids: List[int],
                 batch_size: int,
                 mode: str):
        """Init.

        group_ids: list of group ids (unique integers), one per sequence
        batch_size: size of batches
        mode: 'train' or 'test'
        """
        super().__init__()

        assert mode in ['train', 'test', 'predict']

        self.batch_size = batch_size
        self._mode = mode

        # collect batch info for each group
        self._batch_info = {}
        g_ids_np = np.array(group_ids)
        for group_id in set(group_ids):
            idxs_ = np.where(g_ids_np == group_id)[0]
            # if mode == 'train':
            #     num_batches = len(idxs_) // batch_size # throws away straggler
            # else:
            #     num_batches = int(math.ceil(len(idxs_) / batch_size))
            # if num_batches == 0:
            #     continue
            self._batch_info[group_id] = {'idxs_in_dataset': idxs_}#, 'num_batches': num_batches}

        # [self._batch_info[group_id]['num_batches'] for group_id in self._batch_info.keys()] # batch counts in groups
        # self._num_batches_total = sum([g['num_batches'] for g in self._batch_info.values()])
        self._num_batches_tot = len(self._make_batches())

    def __len__(self) -> int:
        return self._num_batches_tot

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._make_batches()
        for i in np.random.permutation(len(batches)):
            yield batches[i]

    def _make_batches(self):
        """Make a new set of batches. Always the same number of batches."""
        batches = []
        for group_id, d in self._batch_info.items():
            num_idxs = len(d['idxs_in_dataset'])
            if self._mode == 'train':  # permute and truncate list of indices to get integer number of full batches
                num_batches_full = num_idxs // self.batch_size
                if num_batches_full == 0:
                    continue
                num_idxs_tot = self.batch_size * num_batches_full
                idxs_perm = np.random.permutation(d['idxs_in_dataset'])[:num_idxs_tot]
                batches_group = idxs_perm.reshape(num_batches_full, self.batch_size).tolist()
                batches += batches_group
            elif self._mode in ['test', 'predict']: # don't permute (shuffle) and include all samples
                num_batches = int(math.ceil(num_idxs / self.batch_size))
                for i in range(num_batches):
                    batches_group = d['idxs_in_dataset'][i * self.batch_size:(i + 1) * self.batch_size].tolist()
                    assert len(batches_group) > 0
                    batches.append(batches_group)
            else:
                raise NotImplementedError

        return batches


