"""Neural-network models"""

from typing import Union, Tuple, List, Dict
import math
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ytpa_utils.val_utils import is_subset

from src.ml.ml_request import MLRequest
from src.crawler.crawler.constants import COL_VIDEO_ID
from src.ml.model_utils import YTStatsDataset, VariableLengthSequenceBatchSampler, perform_train_epoch, perform_test_epoch
from src.ml.ml_constants import SEQ_LEN_GROUP_WIDTH, COL_SEQ_LEN_ORIG, COL_SEQ_LEN_GROUP, TRAIN_BATCH_SIZE



"""
Tensor of feature vector sequences:
- axis 0: sample index (0, ..., L-1)
- axis 1: sequence index (0, ..., N-1)
- axis 2: feature index (0, ..., D-1)
"""



MODEL_OPTS = dict(
    num_layers_rnn=2,
    hidden_size_rnn=30,
    input_size_rnn=4,
    output_size_rnn=4,
    num_units_embed=[100, 10]
)
TRAIN_OPTS = dict(
    num_epochs=200,
    max_grad_norm=3.0
)




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
        self._dataloader_train = self._prepare_dataloader(data_nonbow, data_bow)
        preproc_params = self._init_preprocessor(self._dataloader_train)

        self._model = Seq2Seq(MODEL_OPTS, preproc_params)

        self._prepare_train_modules()

        # run training epochs
        self._train()

    @staticmethod
    def _prepare_dataloader(data_nonbow: pd.DataFrame,
                            data_bow: pd.DataFrame) \
            -> DataLoader:
        """Make dataset and dataloader for a training run"""
        # dataset
        dataset = YTStatsDataset(data_nonbow, data_bow)

        # sampler
        group_ids = list(dataset._seq_info[COL_SEQ_LEN_GROUP])
        sampler = VariableLengthSequenceBatchSampler(group_ids, TRAIN_BATCH_SIZE)

        # dataloader
        # TODO: adapt to dataloader generating input-output pairs
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

    @staticmethod
    def _init_preprocessor(dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Fit standard normalization preprocessor with one pass through dataset"""
        preproc_params = dict(mu_stats=None, std_stats=None, mu_embeds=None, std_embeds=None)

        # accumulate stats with one pass over dataset
        xs: Dict[str, np.ndarray] = dict(stats=None, embeds=None)
        xsq: Dict[str, np.ndarray] = dict(stats=None, embeds=None)
        count = dict(stats=0, embeds=0)

        for i, data in enumerate(dataloader):
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

        return preproc_params

    def _prepare_train_modules(self):
        """Set up training modules, hyperparameters, etc."""
        LEARNING_RATE = 0.1
        MOMENTUM = 0.3
        DAMPENING = 0.0
        WEIGHT_DECAY = 1e-8

        self._num_batches_train = len(self._dataloader)

        # set up loss function
        self._loss_fn = nn.MSELoss()

        # set up optimizer
        self._optimizer = optim.SGD(self._model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING,
                                    weight_decay=WEIGHT_DECAY)

        # set up scheduler
        self._scheduler = None
        # if SCHEDULER == 'MultiStepLR':
        #     milestones = [int(0.5 * num_epochs), int(0.8 * num_epochs)]
        #     self._scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    def _train(self):
        """Training procedure"""
        num_epochs = TRAIN_OPTS['num_epochs']
        max_grad_norm = TRAIN_OPTS['max_grad_norm']

        # train
        print("===== Training =====")

        train_losses = []
        test_losses = []

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
            # if eval_test:
            #     t11 = time.time()
            #     test_loss_ = perform_test_epoch(model, loss_fn, dataloader_test)
            #     test_losses.append(test_loss_)
            #     print(f'Time elapsed in test epoch: {int(time.time() - t11)} seconds.')

            # update rate scheduler
            if self._scheduler is not None:
                self._scheduler.step()

            # print
            train_abs_err = math.sqrt(train_losses[epoch] / (self._num_batches_train))
            print(f' epoch: {epoch + 1}, train loss: {train_losses[epoch]}, train abs err: {train_abs_err}')
            # if eval_test:
            #     test_abs_err = math.sqrt(test_losses[epoch] / (num_batches_test))
            #     print(f' epoch: {epoch + 1}, test loss: {test_losses[epoch]}, test abs err: {test_abs_err}'

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
    def __init__(self,
                 model_opts: dict,
                 preproc_params: Dict[str, np.ndarray]):
        super().__init__()

        # model config specific to recurrent structure
        self._num_layers_rnn: int = model_opts['num_layers_rnn']
        self._hidden_size_rnn: int = model_opts['hidden_size_rnn']
        self._input_size_rnn: int = model_opts['input_size_rnn']
        self._output_size_rnn: int = model_opts['output_size_rnn']

        # model config specific to embedding structure
        self._num_units_embed: List[int] = model_opts['num_units_embed']
        self._num_layers_embed = len(self._num_units_embed) - 1
        self._num_embedded_features = self._num_units_embed[-1]
        self._input_size_rnn_merged = self._input_size_rnn + self._num_embedded_features
        self._decoder_input_size = 1  # self._input_size_rnn_merged

        # model definition for embedding
        for i in range(self._num_layers_embed):
            self.__dict__['embed_linear' + str(i)] = nn.Linear(
                self._num_units_embed[i],
                self._num_units_embed[i + 1]
            )
        self.embed_dropout = nn.Dropout(0.2)

        # model definition for recurrence
        self.encoder = nn.LSTM(
            self._input_size_rnn_merged,
            self._hidden_size_rnn,
            num_layers=self._num_layers_rnn,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            self._decoder_input_size,
            self._hidden_size_rnn,
            num_layers=self._num_layers_rnn,
            batch_first=True
        )
        self.h2o = nn.Linear(self._hidden_size_rnn, self._output_size_rnn)

        # preprocessing
        assert set(preproc_params) == {'mu_stats', 'std_stats', 'mu_embeds', 'std_embeds'}
        assert np.all(preproc_params['std_stats'] > 1e-5)
        assert np.all(preproc_params['std_embeds'] > 1e-5)
        self._preproc_params = dict(
            mu_stats=torch.as_tensor(preproc_params['mu_stats'], dtype=torch.float),#, device=self._device),
            std_stats=torch.as_tensor(preproc_params['std_stats'], dtype=torch.float),#, device=self._device)
            mu_embeds=torch.as_tensor(preproc_params['mu_embeds'], dtype=torch.float),#, device=self._device),
            std_embeds=torch.as_tensor(preproc_params['std_embeds'], dtype=torch.float)#, device=self._device),
        )

        # reset hidden state
        self._hidden = None

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
        num_predict = data['num_predict']

        assert x_stats.ndim == 3
        assert x_embeds.ndim == 1
        assert x_stats.shape[-1] == self._input_size_rnn
        assert x_embeds.shape[-1] == self._num_units_embed[0]

        # initialize hidden
        self._init_zero_hidden(x_stats.shape[0])

        # preprocess
        x_stats, x_embeds = self._preprocess(x_stats, x_embeds)

        # encode and decode
        self._encode(x_stats, x_embeds)
        out = self._decode(num_predict)

        return out

    def _encode(self,
                x_stats: torch.Tensor,
                x_embeds: torch.Tensor):
        """Run encoder"""
        # process embeddings
        x_embeds = self._process_embeddings(x_embeds)

        # concatenate stats and compressed embeddings
        x_embeds = x_embeds[:, None, :]
        x_embeds = x_embeds.repeat(1, x_stats.shape[1], 1)
        x_merge = torch.concat((x_stats, x_embeds), dim=2)

        # process merged features
        _, hidden_new = self.encoder(x_merge, self._hidden)
        self._update_hidden(hidden_new)

    def _decode(self, num_predict: int) -> torch.Tensor:
        """Run decoder"""
        # recurrence
        batch_size = self._hidden[0].shape[1]
        x_decode = torch.zeros((batch_size, num_predict, self._decoder_input_size))
        out, hidden_new = self.encoder(x_decode, self._hidden)
        self._update_hidden(hidden_new)

        # output layer
        out = self.h2o(out)

        return out

    def _update_hidden(self, hidden_new):
        """Set hidden var with new value"""
        self._hidden = (hidden_new[0].detach().clone(), hidden_new[1].detach().clone())

    def _preprocess(self,
                    x_stats: torch.Tensor,
                    x_embeds: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply preprocessing to feature vectors. Operates on last (feature) dim."""
        x_stats = (x_stats - self._preproc_params['mu_stats']) / self._preproc_params['std_stats']
        x_embeds = (x_embeds - self._preproc_params['mu_embeds']) / self._preproc_params['std_embeds']

        return x_stats, x_embeds

    def _process_embeddings(self, x_embeds: torch.Tensor) -> torch.Tensor:
        """Process embeddings"""
        for i in range(self._num_layers_embed):
            lyr = self.__dict__['embed_linear' + str(i)]
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
        hidden = torch.zeros(self._num_layers_rnn, batch_size, self._hidden_size_rnn, requires_grad=False)#, device=self._device)
        cell = torch.zeros(self._num_layers_rnn, batch_size, self._hidden_size_rnn, requires_grad=False)#, device=self._device)
        self._hidden = (hidden, cell)

