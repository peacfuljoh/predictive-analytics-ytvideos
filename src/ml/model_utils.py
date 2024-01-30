"""Utils for model fit and predict internals"""

import os
from typing import Tuple, List, Dict, Iterator, Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, Sampler
from torch import optim, nn
from torch.utils.data import DataLoader

from ytpa_utils.io_utils import load_pickle
from ytpa_utils.df_utils import join_on_dfs, convert_mixed_df_to_array

from src.crawler.crawler.constants import (KEYS_TRAIN_ID, FEATURES_VECTOR_COL, KEYS_FOR_FIT_NONBOW_SRC,
                                           KEYS_FOR_FIT_NONBOW_TGT, KEYS_TRAIN_NUM, COL_VIDEO_ID,
                                           COL_TIMESTAMP_ACCESSED)
from src.ml.ml_constants import KEYS_EXTRACT_SEQ2SEQ, SEQ_LEN_GROUP_WIDTH, COL_SEQ_LEN_ORIG, COL_SEQ_INFO_GROUP_ID, \
    COL_SEQ_LEN_GROUP, MIN_SAMP_SWITCH_FRAC, MAX_SAMP_SWITCH_FRAC
# from src.rnnoise.constants import DEVICE, MODELS_DIR, MODEL_TYPES


def _dfs_to_arrs_with_src_tgt(data_bow: pd.DataFrame,
                              data_nonbow: pd.DataFrame,
                              mode: str) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Join dataframes on IDs and construct input and output arrays. Some stats are given _src and _tgt suffixes."""
    assert mode in ['train', 'test']

    df_join = join_on_dfs(data_nonbow, data_bow, KEYS_TRAIN_ID, df1_keys_select=[FEATURES_VECTOR_COL])
    X = convert_mixed_df_to_array(df_join, [FEATURES_VECTOR_COL] + KEYS_FOR_FIT_NONBOW_SRC)  # order matters - DO NOT change order
    y = None if mode == 'test' else convert_mixed_df_to_array(df_join, KEYS_FOR_FIT_NONBOW_TGT)

    return X, y

def _dfs_to_arr_seq(data_bow: pd.DataFrame,
                    data_nonbow: pd.DataFrame) \
        -> np.ndarray:
    """Join dataframes on IDs and construct input and output arrays."""
    df_join = join_on_dfs(data_nonbow, data_bow, KEYS_TRAIN_ID, df1_keys_select=[FEATURES_VECTOR_COL])
    X = convert_mixed_df_to_array(df_join, KEYS_TRAIN_NUM + [FEATURES_VECTOR_COL])

    return X

def _dfs_to_train_seqs(data_bow: pd.DataFrame,
                       data_nonbow: pd.DataFrame) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Get time-independent and -dependent data in arrays. Rows of data_nonbow are assumed to be pre-sorted."""
    video_id = data_nonbow.iloc[0][COL_VIDEO_ID]
    assert (data_nonbow[COL_VIDEO_ID] == video_id).all() # ensure that only one video is being processed

    arr_td = data_nonbow[KEYS_TRAIN_NUM].to_numpy() # time-dependent 2D array
    df_ti = data_bow[data_bow[COL_VIDEO_ID] == video_id].iloc[0][FEATURES_VECTOR_COL]
    arr_ti = np.array(list(df_ti)) # time-independent 1D array

    return arr_td, arr_ti




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

        num_samps = input_['stats'].shape[0]
        samp_switch = np.random.randint(int(num_samps * MIN_SAMP_SWITCH_FRAC), int(num_samps * MAX_SAMP_SWITCH_FRAC))

        sample = {
            'stats': torch.tensor(input_['stats'][:samp_switch]),
            'embeds': torch.tensor(input_['embeds']),
            'target': torch.tensor(input_['stats'][samp_switch:])
        }

        return sample


class VariableLengthSequenceBatchSampler(Sampler[List[int]]):
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






"""Utils for training and evaluating RNN model"""
def perform_train_epoch(model: nn.Module,
                        loss_fn,
                        optimizer: optim.Optimizer,
                        dataloader_train: DataLoader,
                        device: Optional[int] = None,
                        max_grad_norm: Optional[float] = None) \
        -> float:
    """Perform one training epoch"""
    # if device is None:
    #     device = DEVICE

    is_training = model.training
    model.train()

    epoch_losses = []
    for i, (X, Y) in enumerate(dataloader_train):
        # if X.device != device and Y.device != device:
        #     X, Y = X.to(device), Y.to(device)
        model.zero_grad(set_to_none=True)
        _, loss = forward_pass_through_seq(model, X, Y=Y, loss_fn=loss_fn, return_Y_pred=False)#, device=device)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        epoch_losses.append(loss.detach().item())

    train_loss = torch.tensor(epoch_losses).sum().item()

    model.training = is_training

    return train_loss


@torch.no_grad()
def perform_test_epoch(model: nn.Module,
                       loss_fn,
                       dataloader_test: DataLoader,
                       device: Optional[int] = None) \
        -> float:
    """Evaluate trained RNNoise denoisers on a test dataset"""
    # if device is None:
    #     device = DEVICE

    is_training = model.training
    model.eval()

    epoch_losses = []
    for i, (X, Y) in enumerate(dataloader_test):
        # if X.device != device and Y.device != device:
        #     X, Y = X.to(device), Y.to(device)
        _, loss = forward_pass_through_seq(model, X, Y=Y, loss_fn=loss_fn, return_Y_pred=False)#, device=device)

        epoch_losses.append(loss.detach().item())

    test_loss = torch.tensor(epoch_losses).sum().item()

    model.training = is_training

    return test_loss


def forward_pass_through_seq(model: nn.Module,
                             X: torch.Tensor,
                             Y: Optional[torch.Tensor] = None,
                             loss_fn = None,
                             return_Y_pred: bool = True,
                             device: Optional[int] = None) \
        -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Full forward pass through sequence, accumulating loss along the way."""
    # if device is None:
    #     device = DEVICE

    # assert model._type in MODEL_TYPES

    batch_size, num_frames, _ = X.shape
    if Y is not None:
        assert Y.shape == (batch_size, num_frames, model._output_size)

    # perform forward pass, save outputs and loss
    loss = torch.zeros(1, dtype=torch.float)#, device=device)
    model.reset_hidden()

    # process
    out = model(X)
    if Y is not None and loss_fn is not None:
        loss = loss_fn(out, Y)
    Y_pred = out.detach().clone() if return_Y_pred else None

    return Y_pred, loss


# def load_model(model_id: str,
#                epoch: Optional[int] = None) \
#         -> [nn.Module, str, str, str]:
#     """Load neural network model from model store"""
#     model_subdir = os.path.join(MODELS_DIR, model_id)
#     model_fnames = [fname for fname in sorted(os.listdir(model_subdir)) if '.pickle' in fname and 'meta' not in fname]
#
#     if epoch is None: # last snapshot
#         model_fname = model_fnames[-1]
#     else: # get closest one
#         epoch_nums = np.array([int(fname[:3]) for fname in model_fnames])
#         idx_ = np.argmin(np.abs(epoch_nums - epoch))
#         model_fname = model_fnames[idx_]
#
#     model_fpath = os.path.join(model_subdir, model_fname)
#     model_obj = load_pickle(model_fpath)
#
#     model = model_obj['model']
#
#     return model, model_fname, model_fpath, model_subdir