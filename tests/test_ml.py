"""Testing for ML models"""

import copy

import pandas as pd
import numpy as np

from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom
from ytpa_utils.df_utils import join_on_dfs
from src.crawler.crawler.constants import FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS, \
    ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM, ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_SR_ALPHAS



def test_rand_lin_proj():
    pass

    # # params
    # num_samps = 20
    # num_features = 100
    # num_nonzero_per_samp = 5
    # num_dims_embed = 10
    # density = 0.5
    # sr_alpha = 0.0001
    #
    # num_nonzero_in_proj = (num_dims_embed * num_features) * density
    #
    # # raw_data
    # data = {FEATURES_VECTOR_COL: []}
    # for _ in range(num_samps):
    #     idxs = np.random.permutation(num_features)[:num_nonzero_per_samp]
    #     vals = np.random.randint(0, 20, num_nonzero_per_samp)
    #     data[FEATURES_VECTOR_COL].append([pt for pt in zip(idxs, vals)])
    # df = pd.DataFrame(data)
    #
    # # base config
    # config = {
    #     ML_MODEL_TYPE: ML_MODEL_TYPE_LIN_PROJ_RAND,
    #     ML_MODEL_HYPERPARAMS: {
    #         ML_HYPERPARAM_EMBED_DIM: num_dims_embed,
    #         ML_HYPERPARAM_RLP_DENSITY: density,
    #         ML_HYPERPARAM_SR_ALPHAS: sr_alpha
    #     }
    # }
    #
    # #
    # ml_request = MLRequest(config)
    # model = MLModelLinProjRandom(ml_request)
    # model.fit(df)
    #
    # assert model.transform(df).shape == (num_samps, num_dims_embed)

