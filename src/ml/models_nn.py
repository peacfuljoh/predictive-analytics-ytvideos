"""Neural-network models"""

import os
from typing import Union, Tuple, List, Dict, Optional
import math
import time
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from ytpa_utils.val_utils import is_subset
from ytpa_utils.io_utils import save_pickle

from src.ml.ml_request import MLRequest
from src.ml.torch_utils import perform_train_epoch, perform_test_epoch
from src.ml.datasets import YTStatsDataset, VariableLengthSequenceBatchSampler
from src.ml.ml_constants import (SEQ_LEN_GROUP_WIDTH, COL_SEQ_LEN_ORIG, COL_SEQ_LEN_GROUP, TRAIN_BATCH_SIZE,
                                 NUM_EPOCHS_PER_SNAPSHOT)
from src.crawler.crawler.constants import (COL_VIDEO_ID, TRAIN_SEQ_PERIOD, VEC_EMBED_DIMS, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, KEYS_TRAIN_NUM, KEYS_TRAIN_NUM_TGT_IDXS,
                                           VEC_EMBED_DIMS_NN, KEYS_TRAIN_NUM_TGT)



"""
Tensor of feature vector sequences:
- axis 0: sample index (0, ..., L-1)
- axis 1: sequence index (0, ..., N-1)
- axis 2: feature index (0, ..., D-1)
"""


USE_EMBEDDINGS = False




MODEL_OPTS = dict(
    num_layers_rnn=2,
    num_rnn_blocks=6,
    hidden_size_rnn=30,
    input_size_rnn=len(KEYS_TRAIN_NUM_TGT_IDXS),
    output_size_rnn=len(KEYS_TRAIN_NUM_TGT_IDXS),
    num_units_embed=[VEC_EMBED_DIMS_NN, 100, 10]
)
TRAIN_OPTS = dict(
    num_epochs=100,
    lr=0.05,
    momentum=0.3,
    max_grad_norm=3.0,
    scheduler='None'#'MultiStepLR'
)



class MLModelSeq2Seq():
    """Wrapper class for handling sequence-to-sequence model train and test"""
    def __init__(self,
                 ml_request: MLRequest,
                 verbose: bool = False,
                 model_dir: Optional[str] = None,
                 metadata: Optional[dict] = None,
                 model: Optional[nn.Module] = None):
        self._verbose = verbose

        # fields
        self._config = None
        self._model = model
        self._model_dir: Optional[str] = model_dir
        self._metadata: Optional[dict] = metadata

        self._dataloader_train = None
        self._dataloader_test = None

        # validate and store config
        assert ml_request.get_valid()
        self._config = ml_request.get_config()

    def fit(self,
            data_nonbow: pd.DataFrame,
            data_bow: pd.DataFrame,
            data_nonbow_test: Optional[pd.DataFrame] = None):
        """
        Fit the model with static (bow) and dynamic (nonbow) features.
        """
        # make sure metadata is available for all videos
        assert is_subset(list(data_nonbow[COL_VIDEO_ID]), list(data_bow[COL_VIDEO_ID]))

        self._dt_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # setup
        self._dataloader_train = _prepare_dataloader(data_nonbow, data_bow, 'train')
        if data_nonbow_test is not None:
            self._dataloader_test = _prepare_dataloader(data_nonbow_test, data_bow, 'test')
        preproc_params = _init_preprocessor(self._dataloader_train)

        self._model = Seq2Seq(MODEL_OPTS, preproc_params)

        self._prepare_train_modules()

        # run training epochs
        self._train()

    def _prepare_train_modules(self):
        """Set up training modules, hyperparameters, etc."""
        LEARNING_RATE = TRAIN_OPTS['lr']
        MOMENTUM = TRAIN_OPTS['momentum']
        DAMPENING = 0.0
        WEIGHT_DECAY = 1e-8
        SCHEDULER = TRAIN_OPTS.get('scheduler', 'None')
        NUM_EPOCHS = TRAIN_OPTS['num_epochs']

        self._eval_test = self._dataloader_test is not None

        self._num_batches_train = len(self._dataloader_train)
        self._num_batches_test = max(len(self._dataloader_test), 1) if self._eval_test else 1

        # set up loss function
        self._loss_fn = nn.MSELoss()

        # set up optimizer
        self._optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING,
                                    weight_decay=WEIGHT_DECAY)

        # set up scheduler
        if SCHEDULER == 'None':
            self._scheduler = None
        elif SCHEDULER == 'MultiStepLR':
            milestones = [int(0.5 * NUM_EPOCHS), int(0.8 * NUM_EPOCHS)]
            self._scheduler = MultiStepLR(self._optimizer, milestones=milestones, gamma=0.2)
        else:
            raise NotImplementedError

    def _train(self):
        """Training procedure"""
        num_epochs = TRAIN_OPTS['num_epochs']
        max_grad_norm = TRAIN_OPTS['max_grad_norm']


        # train
        print("===== Training =====")

        train_losses = []
        test_losses = []
        test_loss_ = 0.0

        # iterate
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}')

            # train
            t00 = time.time()
            train_loss_ = perform_train_epoch(self._model, self._loss_fn, self._optimizer, self._dataloader_train,
                                              max_grad_norm=max_grad_norm)
            train_losses.append(train_loss_)
            print(f'Time elapsed in train epoch: {int(time.time() - t00)} seconds.')

            # test
            if self._eval_test:
                t11 = time.time()
                test_loss_ = perform_test_epoch(self._model, self._loss_fn, self._dataloader_test)
                print(f'Time elapsed in test epoch: {int(time.time() - t11)} seconds.')
            test_losses.append(test_loss_)

            # update rate scheduler
            if self._scheduler is not None:
                self._scheduler.step()

            # print
            train_abs_err = math.sqrt(train_losses[epoch] / (self._num_batches_train))
            print(f' epoch: {epoch + 1}, train loss: {train_losses[epoch]}, train abs err: {train_abs_err}')
            if self._eval_test:
                test_abs_err = math.sqrt(test_losses[epoch] / (self._num_batches_test))
                print(f' epoch: {epoch + 1}, test loss: {test_losses[epoch]}, test abs err: {test_abs_err}')

            # save latest info to pickle file
            epoch_to_save = (not np.mod(epoch, NUM_EPOCHS_PER_SNAPSHOT)) or (epoch == num_epochs - 1)
            if (self._model_dir is not None) and epoch_to_save:
                df_info = pd.DataFrame(dict(
                    epoch=list(range(epoch + 1)),
                    loss_train=train_losses,
                    loss_test=test_losses
                ))
                obj = dict(
                    model=self._model,
                    stats=df_info,
                    modules=dict(
                        loss_fn=self._loss_fn,
                        optimizer=self._optimizer,
                        scheduler=self._scheduler
                    ),
                    options=dict(
                        model_opts=MODEL_OPTS,
                        train_opts=TRAIN_OPTS
                    ),
                    metadata=self._metadata,
                    epoch=epoch
                )
                fpath = os.path.join(self._model_dir, self._dt_str + '__' + str(epoch) + '.pickle')
                save_pickle(fpath, obj)

    def predict(self,
                data_nonbow: pd.DataFrame,
                data_bow: pd.DataFrame,
                pred_opts: Optional[dict] = None) \
            -> pd.DataFrame:
        """Feed test data through model"""
        return _predict(data_nonbow, data_bow, pred_opts, self._model)




def _prepare_dataloader(data_nonbow: pd.DataFrame,
                        data_bow: pd.DataFrame,
                        mode: str) \
        -> DataLoader:
    """Make dataset and dataloader for a training run"""
    assert mode in ['train', 'test', 'predict']

    # dataset
    dataset = YTStatsDataset(data_nonbow, data_bow, mode)

    # sampler
    group_ids = list(dataset._seq_info[COL_SEQ_LEN_GROUP])
    sampler = VariableLengthSequenceBatchSampler(group_ids, TRAIN_BATCH_SIZE, mode)

    # dataloader
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=8, pin_memory=True)

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

    return dataloader


def _init_preprocessor(dataloader: DataLoader) -> Dict[str, np.ndarray]:
    """Fit standard normalization preprocessor with one pass through dataset"""
    preproc_params = dict(mu_stats=None, std_stats=None,
                          mu_stats_un=None, std_stats_un=None,
                          mu_embeds=None, std_embeds=None)

    # accumulate stats with one pass over dataset
    xs: Dict[str, np.ndarray] = dict(stats=None, embeds=None)
    xsq: Dict[str, np.ndarray] = dict(stats=None, embeds=None)
    count = dict(stats=0, embeds=0)

    for i, (data, _) in enumerate(dataloader):
        x_stats = data['stats']
        x_embeds = data['embeds']

        if i == 0:
            input_size = x_stats.shape[2]
            xs['stats'] = np.zeros(input_size, dtype='float')
            xsq['stats'] = np.zeros(input_size, dtype='float')

            input_size = x_embeds.shape[1]
            xs['embeds'] = np.zeros(input_size, dtype='float')
            xsq['embeds'] = np.zeros(input_size, dtype='float')

        if np.mod(i, 100) == 0:
            print(f' batch {i}/{len(dataloader)}')

        X_rav_stats = x_stats.reshape(x_stats.shape[0] * x_stats.shape[1], x_stats.shape[2])
        xs['stats'] += X_rav_stats.sum(0).numpy()  # sum over batch and sequence dims, keep feat dim
        xsq['stats'] += (X_rav_stats ** 2).sum(0).numpy()
        count['stats'] += X_rav_stats.shape[0]

        X_rav_embeds = x_embeds
        xs['embeds'] += X_rav_embeds.sum(0).numpy()  # sum over batch and sequence dims, keep feat dim
        xsq['embeds'] += (X_rav_embeds ** 2).sum(0).numpy()
        count['embeds'] += X_rav_embeds.shape[0]

    # compute mean and standard deviation
    for key in ['stats', 'embeds']:
        mu = xs[key] / count[key]
        std = np.sqrt((xsq[key] - 2 * mu * xs[key] + count[key] * mu ** 2) / count[key])
        preproc_params['mu_' + key] = mu
        preproc_params['std_' + key] = std

    # include modified params for unprocessing outputs
    idxs_un = KEYS_TRAIN_NUM_TGT_IDXS
    preproc_params['mu_stats_un'] = preproc_params['mu_stats'][idxs_un]
    preproc_params['std_stats_un'] = preproc_params['std_stats'][idxs_un]

    return preproc_params

def _predict(data_nonbow: pd.DataFrame,
             data_bow: pd.DataFrame,
             pred_opts: dict,
             model: nn.Module) \
        -> pd.DataFrame:
    """Predict video stats into the future"""
    if pred_opts is None:
        pred_opts = {}

    pred_horizon = pred_opts.get('pred_horizon', 7 * 24 * 3600)  # seconds to predict from start of each video's records

    # set up dataloader and model options
    dataloader_pred = _prepare_dataloader(data_nonbow, data_bow, 'predict')

    is_training = model.training
    model.eval()
    model.predicting = True

    # predict
    num_videos_all = len(data_nonbow[COL_VIDEO_ID].unique())
    print(f"Making predictions for {num_videos_all} videos.")

    df_out = []
    for n, (X, _) in enumerate(dataloader_pred):
        # reset hidden state
        model.reset_hidden()

        # determine number of steps to predict
        num_videos, num_steps_data, _ = X['stats'].shape
        num_steps_pred_tot = int(pred_horizon / TRAIN_SEQ_PERIOD)
        num_steps_pred = num_steps_pred_tot - num_steps_data
        if num_steps_pred <= 0:
            continue
        X['num_predict'] = num_steps_pred

        # make prediction
        out = model(X)

        # pack into output DataFrame
        for b, video_id in enumerate(X['video_id']):
            preds = out[b, :, :].detach().numpy() # time steps x features
            df_ = pd.DataFrame(preds, columns=KEYS_TRAIN_NUM_TGT)
            df_[COL_USERNAME] = data_bow.query(f"{COL_VIDEO_ID} == '{video_id}'").iloc[0][COL_USERNAME]
            df_[COL_VIDEO_ID] = video_id
            ts_last = data_nonbow.query(f"{COL_VIDEO_ID} == '{video_id}'")[COL_TIMESTAMP_ACCESSED].max()
            ts_pred = [ts_last + datetime.timedelta(seconds=(i + 1) * TRAIN_SEQ_PERIOD)
                for i in range(num_steps_pred)]
            df_[COL_TIMESTAMP_ACCESSED] = ts_pred
            df_out.append(df_)

    model.training = is_training
    model.predicting = False

    df_out = pd.concat(df_out, axis=0, ignore_index=True)

    return df_out













class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence recurrent model.

    Ref: https://arxiv.org/abs/1409.3215
    """
    def __init__(self,
                 model_opts: dict,
                 preproc_params: Dict[str, np.ndarray]):
        super().__init__()

        # model config specific to recurrent structure
        self._num_rnn_blocks: int = model_opts['num_rnn_blocks']
        self._num_layers_rnn: int = model_opts['num_layers_rnn']
        self._hidden_size_rnn: int = model_opts['hidden_size_rnn']
        self._input_size_rnn: int = model_opts['input_size_rnn']
        self._output_size_rnn: int = model_opts['output_size_rnn']

        # model config specific to embedding structure
        self._num_units_embed: List[int] = model_opts['num_units_embed']
        self._num_layers_embed = len(self._num_units_embed) - 1
        self._num_embedded_features = self._num_units_embed[-1]
        if USE_EMBEDDINGS:
            self._input_size_rnn_merged = self._input_size_rnn + self._num_embedded_features
        else:
            self._input_size_rnn_merged = self._input_size_rnn
        # self._decoder_input_size = 1  # self._input_size_rnn_merged

        # model definition for embedding
        for i in range(self._num_layers_embed):
            lyr = nn.Linear(self._num_units_embed[i], self._num_units_embed[i + 1])
            setattr(self, 'embed_linear' + str(i), lyr)
        dropout = model_opts.get('dropout', 0.2)
        self.embed_dropout = nn.Dropout(dropout)

        # model definition for recurrence
        self.i2h = nn.Linear(self._input_size_rnn_merged, self._hidden_size_rnn)
        for i in range(self._num_rnn_blocks):
            setattr(self, f"rnn{i + 1}", make_lstm_block(self._hidden_size_rnn, self._num_layers_rnn))

        # output layer
        self.h2o = nn.Linear(self._hidden_size_rnn, self._output_size_rnn)

        # preprocessing
        assert set(preproc_params) == {'mu_stats', 'std_stats',
                                       'mu_stats_un', 'std_stats_un',
                                       'mu_embeds', 'std_embeds'}
        assert np.all(preproc_params['std_stats'] > 1e-5)
        assert np.all(preproc_params['std_embeds'] > 1e-5)
        self._preproc_params = dict(
            mu_stats=torch.tensor(preproc_params['mu_stats'], dtype=torch.float, requires_grad=False),#, device=self._device),
            std_stats=torch.tensor(preproc_params['std_stats'], dtype=torch.float, requires_grad=False),#, device=self._device)
            mu_stats_un=torch.tensor(preproc_params['mu_stats_un'], dtype=torch.float, requires_grad=False),#, device=self._device),
            std_stats_un=torch.tensor(preproc_params['std_stats_un'], dtype=torch.float, requires_grad=False),#, device=self._device)
            mu_embeds=torch.tensor(preproc_params['mu_embeds'], dtype=torch.float, requires_grad=False),#, device=self._device),
            std_embeds=torch.tensor(preproc_params['std_embeds'], dtype=torch.float, requires_grad=False)#, device=self._device),
        )

        # init
        self._hidden = None
        self._steps_pred = None
        self.predicting = False

    def forward(self, data: Dict[str, Union[torch.Tensor, int]]) -> torch.Tensor:
        """
        Forward pass through network for a batch.

        "data" contains a batch of stats time series and the corresponding embedding feature vector:
            data['stats'], data['embeds']
        It also contains a number of future steps to predict:
            data['num_predict']

        data['stats']: batch_size x seq_len x num_input_features
        out: batch_size x seq_len x num_output_features
        """
        x_stats = data['stats']
        x_embeds = data['embeds']

        if self.predicting:
            num_predict = data['num_predict']
            assert num_predict >= 1

        assert not (self.training and self.predicting)
        assert x_stats.ndim == 3
        assert x_embeds.ndim == 2
        assert x_stats.shape[-1] == self._input_size_rnn
        assert x_embeds.shape[-1] == self._num_units_embed[0]

        # initialize hidden
        self._init_zero_hidden(x_stats.shape[0])

        # preprocess
        x_stats, x_embeds = self._preprocess(x_stats, x_embeds=x_embeds)

        # encode and decode
        if USE_EMBEDDINGS:
            x_embeds = self._process_embeddings(x_embeds)
        out = self._process_inputs(x_stats, x_embeds)
        if self.predicting:
            out = self._predict(out[:, -1:, :], x_embeds, num_predict)
            out = self._unpreprocess(out)

        return out

    def _process_inputs(self,
                        x_stats: torch.Tensor,
                        x_embeds: torch.Tensor) \
            -> torch.Tensor:
        """Feed preprocessed inputs through network"""
        if USE_EMBEDDINGS:
            x_merge = self._merge_stats_embeds(x_stats, x_embeds)
        else:
            x_merge = x_stats
        x_out = self._feed_to_rnn(x_merge)
        x_out = self.h2o(x_out)

        return x_out

    def _predict(self,
                 x_out: torch.Tensor,
                 x_embeds: torch.Tensor,
                 num_predict: int) \
            -> torch.Tensor:
        """Predict future values"""
        out = torch.zeros((x_out.shape[0], num_predict, x_out.shape[2]), dtype=torch.float)
        out[:, 0, :] = x_out[:, 0, :]

        for i in range(num_predict - 1):
            x_out = self._process_inputs(x_out, x_embeds)
            out[:, i + 1, :] = x_out[:, 0, :]

        return out

    def _merge_stats_embeds(self,
                            x_stats: torch.Tensor,
                            x_embeds: torch.Tensor) \
            -> torch.Tensor:
        """Merge stats tensor with embeddings tensor"""
        x_embeds = x_embeds[:, None, :]
        x_embeds = x_embeds.repeat(1, x_stats.shape[1], 1)
        x_merge = torch.concat((x_stats, x_embeds), dim=2)

        return x_merge

    def _feed_to_rnn(self, input_: torch.Tensor) -> torch.Tensor:
        """Feed tensor to RNN model"""
        out = self.i2h(input_)
        # out = self.embed_dropout(out) # try this
        for i in range(self._num_rnn_blocks):
            nn_module = getattr(self, f"rnn{i + 1}")
            out_new, self._hidden[i] = apply_module_with_hidden(nn_module, out, self._hidden[i])
            if i < self._num_rnn_blocks - 1:
                out = out + out_new
            else:
                out = out_new

        return out

    def _preprocess(self,
                    x_stats: torch.Tensor,
                    x_embeds: Optional[torch.Tensor] = None,
                    mode: str = 'inputs') \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply preprocessing to feature vectors. Operates on last (feature) dim."""
        assert mode in ['inputs', 'outputs']

        if mode == 'inputs':
            x_stats = (x_stats - self._preproc_params['mu_stats']) / self._preproc_params['std_stats']
        elif mode == 'outputs':
            x_stats = (x_stats - self._preproc_params['mu_stats_un']) / self._preproc_params['std_stats_un']
        else:
            raise NotImplementedError

        if x_embeds is not None:
            if USE_EMBEDDINGS:
                x_embeds = (x_embeds - self._preproc_params['mu_embeds']) / self._preproc_params['std_embeds']
                return x_stats, x_embeds
            else:
                return x_stats, None
        else:
            return x_stats

    def _unpreprocess(self,
                      x_stats: torch.Tensor,
                      x_embeds: Optional[torch.Tensor] = None) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Undo effect of preprocessing."""
        x_stats = x_stats * self._preproc_params['std_stats_un'] + self._preproc_params['mu_stats_un']
        if x_embeds is not None:
            x_embeds = x_embeds * self._preproc_params['std_embeds_un'] + self._preproc_params['mu_embeds_un']
            return x_stats, x_embeds
        else:
            return x_stats

    def _process_embeddings(self, x_embeds: torch.Tensor) -> torch.Tensor:
        """Process embeddings"""
        for i in range(self._num_layers_embed):
            lyr = getattr(self, 'embed_linear' + str(i))
            x_embeds = lyr(x_embeds)
            # if i < self._num_layers_embed - 1:
            x_embeds = F.relu(x_embeds)
            x_embeds = self.embed_dropout(x_embeds)
        return x_embeds

    def reset_hidden(self):
        """Reset hidden state."""
        self._hidden = None

    def _init_zero_hidden(self, batch_size: int):
        """Returns a hidden state with specified batch size."""
        self._hidden = [make_cell_states(self._num_layers_rnn, batch_size, self._hidden_size_rnn)
                        for _ in range(self._num_rnn_blocks)]







### Helper methods ###
def make_lstm_block(hidden_size_rnn: int,
                    num_layers_rnn: int) \
        -> nn.Module:
    """Make a deep LSTM block"""
    rnn = nn.LSTM(
        hidden_size_rnn,
        hidden_size_rnn,
        num_layers=num_layers_rnn,
        batch_first=True
    )
    return rnn

def make_cell_states(num_layers_rnn: int,
                     batch_size: int,
                     hidden_size_rnn: int) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """Make LSTM hidden and cell state tensors"""
    hidden = torch.zeros(num_layers_rnn, batch_size, hidden_size_rnn, requires_grad=False)  # , device=self._device)
    cell = torch.zeros(num_layers_rnn, batch_size, hidden_size_rnn, requires_grad=False)  # , device=self._device)
    return (hidden, cell)

def apply_module_with_hidden(nn_module: nn.Module,
                             input_: torch.Tensor,
                             hidden: Tuple[torch.Tensor, torch.Tensor]) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Feed tensor to RNN model"""
    out, hidden_new = nn_module(input_, hidden)
    hidden = (hidden_new[0].detach().clone(), hidden_new[1].detach().clone())

    return out, hidden
