"""Machine learning op config"""

import copy

from src.crawler.crawler.constants import (ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS, ML_MODEL_TYPE_LIN_PROJ_RAND,
                                           ML_HYPERPARAM_EMBED_DIM, ML_HYPERPARAM_RLP_DENSITY, ML_CONFIG_KEYS,
                                           ML_MODEL_TYPES, TRAIN_TEST_SPLIT,
                                           TRAIN_TEST_SPLIT_DFLT, ML_HYPERPARAM_SR_ALPHAS, ML_HYPERPARAM_SR_CV_SPLIT,
                                           ML_HYPERPARAM_SR_CV_COUNT, ML_MODEL_TYPE_SEQ2SEQ)
from ytpa_utils.val_utils import is_list_of_floats


class MLRequest():
    """ML process request object. Facilitates config validation and handling."""
    def __init__(self, config: dict):
        self._valid = False

        self._config = self._validate_config(config)

    def _validate_config(self, config: dict) -> dict:
        """Validate all config options"""
        config = copy.deepcopy(config) # to not modify original

        # make sure all mandatory keys are present
        for key in ML_CONFIG_KEYS:
            assert key in config

        # validate model type
        assert config[ML_MODEL_TYPE] in ML_MODEL_TYPES

        # validate by specific model types
        if config[ML_MODEL_TYPE] == ML_MODEL_TYPE_LIN_PROJ_RAND:
            self._validate_lin_proj_rand(config)
        elif config[ML_MODEL_TYPE] == ML_MODEL_TYPE_SEQ2SEQ:
            self._validate_seq2seq(config)
        else:
            raise NotImplementedError(f'The specified model type ({config[ML_MODEL_TYPE]}) is not available.')

        # insert defaults if not present
        config[TRAIN_TEST_SPLIT] = config.get(TRAIN_TEST_SPLIT, TRAIN_TEST_SPLIT_DFLT)

        # mark request as valid
        self._valid = True

        return config

    def _validate_lin_proj_rand(self, config: dict):
        """Validate config options specific to the random linear projection model."""
        # validate hyperparameters
        hp = config[ML_MODEL_HYPERPARAMS]

        # embedding
        assert ML_HYPERPARAM_EMBED_DIM in hp
        assert isinstance(hp[ML_HYPERPARAM_EMBED_DIM], int)

        hp[ML_HYPERPARAM_RLP_DENSITY] = hp.get(ML_HYPERPARAM_RLP_DENSITY, 'auto') # if 'auto': 1 / sqrt(n_features)
        if not isinstance(hp[ML_HYPERPARAM_RLP_DENSITY], str): # must be float
            assert 0.0 < hp[ML_HYPERPARAM_RLP_DENSITY] < 1.0

        # regression
        assert ML_HYPERPARAM_SR_ALPHAS in hp
        assert is_list_of_floats(hp[ML_HYPERPARAM_SR_ALPHAS])

        # cross validation
        assert ML_HYPERPARAM_SR_CV_SPLIT in hp
        assert ML_HYPERPARAM_SR_CV_COUNT in hp

    def _validate_seq2seq(self, config: dict):
        """Validate config options specific to the random linear projection model."""
        # validate hyperparameters
        hp = config[ML_MODEL_HYPERPARAMS]

        # embedding
        assert ML_HYPERPARAM_EMBED_DIM in hp
        assert isinstance(hp[ML_HYPERPARAM_EMBED_DIM], int)

    def get_config(self) -> dict:
        assert self.get_valid()
        return copy.deepcopy(self._config)

    def get_db(self) -> dict:
        assert self.get_valid()
        return self.get_config()['db']

    def get_valid(self) -> bool:
        return self._valid
