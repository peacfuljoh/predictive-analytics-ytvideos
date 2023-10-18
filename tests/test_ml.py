"""Testing for ML models"""

import copy

import pandas as pd
import numpy as np

from src.ml.ml_request import MLRequest
from src.ml.ml_models import MLModelLinProjRandom
from src.crawler.crawler.utils.misc_utils import join_on_dfs
from src.crawler.crawler.constants import FEATURES_VECTOR_COL, ML_MODEL_TYPE, ML_MODEL_HYPERPARAMS, \
    ML_MODEL_TYPE_LIN_PROJ_RAND, ML_HYPERPARAM_EMBED_DIM, ML_HYPERPARAM_RLP_DENSITY, ML_HYPERPARAM_SR_ALPHAS



def test_rand_lin_proj():
    # params
    num_samps = 20
    num_features = 100
    num_nonzero_per_samp = 5
    num_dims_embed = 10
    density = 0.5
    sr_alpha = 0.0001

    num_nonzero_in_proj = (num_dims_embed * num_features) * density

    # raw_data
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
            ML_HYPERPARAM_SR_ALPHAS: sr_alpha
        }
    }

    #
    ml_request = MLRequest(config)
    model = MLModelLinProjRandom(ml_request)
    model.fit(df)

    assert model.transform(df).shape == (num_samps, num_dims_embed)

def test_join_on_dfs():
    # single index
    df0 = pd.DataFrame(
        dict(
            id0=['id_0', 'id_1', 'id_0', 'id_0', 'id_1'],
            val0=[1, 2, 3, 4, 5],
            val1=[6, 7, 8, 9, 10],
            str0=['a', 'b', 'c', 'd', 'e']
        )
    )

    df1 = pd.DataFrame(
        dict(
            id0=['id_0', 'id_1'],
            arr0=[[7, 8, 7], [112, 332, 231]]
        )
    )

    df_join = join_on_dfs(
        df0,
        df1,
        index_keys=['id0'],
        df0_keys_select=['id0', 'val0', 'str0'],
        df1_keys_select=['arr0']
    )

    df_join_expected = pd.DataFrame(
        dict(
            id0=['id_0', 'id_1', 'id_0', 'id_0', 'id_1'],
            val0=[1, 2, 3, 4, 5],
            str0=['a', 'b', 'c', 'd', 'e'],
            arr0=[[7, 8, 7], [112, 332, 231], [7, 8, 7], [7, 8, 7], [112, 332, 231]]
        )
    )

    assert len(df_join) == len(df_join_expected)
    assert all(df_join.columns == df_join_expected.columns)
    for i in df_join.index:
        for key in df_join.columns:
            assert df_join.loc[i, key] == df_join_expected.loc[i, key]

    # multi-index
    df0 = pd.DataFrame(
        dict(
            id0=['id_0', 'id_1', 'id_0', 'id_0', 'id_1'],
            id1=['id_5', 'id_5', 'id_6', 'id_5', 'id_5'],
            val0=[1, 2, 3, 4, 5],
            val1=[6, 7, 8, 9, 10],
            str0=['a', 'b', 'c', 'd', 'e']
        )
    )

    df1 = pd.DataFrame(
        dict(
            id0=['id_0', 'id_0', 'id_1'],
            id1=['id_5', 'id_6', 'id_5'],
            arr0=[[7, 8, 7], [5, 4, 3], [112, 332, 231]]
        )
    )

    df_join = join_on_dfs(
        df0,
        df1,
        index_keys=['id0', 'id1'],
        df0_keys_select=['id0', 'id1', 'val0', 'str0'],
        df1_keys_select=['arr0']
    )

    df_join_expected = pd.DataFrame(
        dict(
            id0=['id_0', 'id_1', 'id_0', 'id_0', 'id_1'],
            id1=['id_5', 'id_5', 'id_6', 'id_5', 'id_5'],
            val0=[1, 2, 3, 4, 5],
            str0=['a', 'b', 'c', 'd', 'e'],
            arr0=[[7, 8, 7], [112, 332, 231], [5, 4, 3], [7, 8, 7], [112, 332, 231]]
        )
    )

    assert len(df_join) == len(df_join_expected)
    assert all(df_join.columns == df_join_expected.columns)
    for i in df_join.index:
        for key in df_join.columns:
            assert df_join.loc[i, key] == df_join_expected.loc[i, key]




if __name__ == '__main__':
    test_rand_lin_proj()
    test_join_on_dfs()
