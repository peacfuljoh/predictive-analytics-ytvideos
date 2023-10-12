"""Testing for ML models"""

import copy

import pandas as pd
import numpy as np

from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom
from src.crawler.crawler.constants import FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS, \
    ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM, ML_HYPERPARAM_RLP_DENSITY



def test_rand_lin_proj():
    # params
    num_samps = 20
    num_features = 100
    num_nonzero_per_samp = 5
    num_dims_embed = 10
    density = 0.5

    num_nonzero_in_proj = (num_dims_embed * num_features) * density

    # data
    data = {FEATURES_VECTOR_COL: []}
    for _ in range(num_samps):
        idxs = np.random.permutation(num_features)[:num_nonzero_per_samp]
        vals = np.random.randint(0, 20, num_nonzero_per_samp)
        data[FEATURES_VECTOR_COL].append([pt for pt in zip(idxs, vals)])
    df = pd.DataFrame(data)

    # base config
    config = {
        ML_MODEL_TYPE: ML_MODEL_TYPE_LIN_PROJ_RAND,
        ML_MODEL_HYPERPARAMS: {
            ML_HYPERPARAM_EMBED_DIM: num_dims_embed,
            ML_HYPERPARAM_RLP_DENSITY: density,
        }
    }

    #
    ml_request = MLRequest(config)
    model = MLModelLinProjRandom(ml_request)
    model.fit(df)

    assert model.transform(df).shape == (num_samps, num_dims_embed)
    # num_nonzeros = len(model._model.components_.nonzero()[0])
    # assert abs(num_nonzero_in_proj - num_nonzeros) < 5 # randomized number of nonzeros in final projector matrix




if __name__ == '__main__':
    test_rand_lin_proj()
