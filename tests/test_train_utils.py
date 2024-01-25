"""Tests for ML train utils"""

import datetime

import pandas as pd
import numpy as np
from pandas import Timestamp

from src.crawler.crawler.constants import COL_VIDEO_ID, COL_USERNAME, COL_TIMESTAMP_ACCESSED
from src.ml.train_utils import resample_to_uniform_grid, train_test_split_uniform, train_test_split_video_id
from src.crawler.crawler.constants import TIMESTAMP_FMT



# TODO: debug (not passing in github actions)
def _test_resample_to_uniform_grid():
    datetime_ = datetime.datetime.strptime('2020-05-05 14:43:11.342435', TIMESTAMP_FMT)
    dts = ([datetime_ + datetime.timedelta(hours=1.55 * i) for i in range(3)] +
           [datetime_ + datetime.timedelta(hours=1.27 * i) for i in range(4)])
    df_in = pd.DataFrame({
        COL_USERNAME: ['a', 'a', 'a', 'b', 'b', 'b', 'b'],
        COL_VIDEO_ID: ['asdf', 'asdf', 'asdf', 'uytr', 'uytr', 'uytr', 'uytr'],
        COL_TIMESTAMP_ACCESSED: dts,
        'vals': [0, 1, 2, 6, 7, 8, 9]
    })

    df_out = resample_to_uniform_grid(df_in, period=3600)

    # print(df_out.to_dict('records'))

    df_expected = [
        {'vals': 0.0000000000000000, 'timestamp_accessed': Timestamp('2020-05-05 14:43:11.342435'), 'username': 'a', 'video_id': 'asdf'},
        {'vals': 0.6451612903225806, 'timestamp_accessed': Timestamp('2020-05-05 15:43:11.342435'), 'username': 'a', 'video_id': 'asdf'},
        {'vals': 1.2903225806451613, 'timestamp_accessed': Timestamp('2020-05-05 16:43:11.342435'), 'username': 'a', 'video_id': 'asdf'},
        {'vals': 1.9354838709677420, 'timestamp_accessed': Timestamp('2020-05-05 17:43:11.342435'), 'username': 'a', 'video_id': 'asdf'},
        {'vals': 6.0000000000000000, 'timestamp_accessed': Timestamp('2020-05-05 14:43:11.342435'), 'username': 'b', 'video_id': 'uytr'},
        {'vals': 6.7874015748031480, 'timestamp_accessed': Timestamp('2020-05-05 15:43:11.342435'), 'username': 'b', 'video_id': 'uytr'},
        {'vals': 7.5748031496062970, 'timestamp_accessed': Timestamp('2020-05-05 16:43:11.342435'), 'username': 'b', 'video_id': 'uytr'},
        {'vals': 8.3622047244094480, 'timestamp_accessed': Timestamp('2020-05-05 17:43:11.342435'), 'username': 'b', 'video_id': 'uytr'}
    ]
    df_expected = pd.DataFrame(df_expected)

    assert df_out.equals(df_expected)


def test_train_test_split():
    np.random.seed(10)

    # setup
    df_ = pd.DataFrame({
        COL_USERNAME: ['a'] * 3 + ['a'] * 5 + ['a'] * 2 + ['b'] * 4 + ['b'] * 3,
        COL_VIDEO_ID: ['asdf'] * 3 + ['wwee'] * 5 + ['fsfs'] * 2 + ['blah'] * 4 + ['mlem'] * 3,
        'vals': list(range(17))
    })
    num_samps = df_.shape[0]
    data = dict(nonbow=df_)

    tt_split = 0.8

    # uniform split
    data_nonbow_train, data_nonbow_test = train_test_split_uniform(data, tt_split)

    num_train = int(tt_split * num_samps)
    assert data_nonbow_train.shape[0] == num_train
    assert data_nonbow_test.shape[0] == num_samps - num_train

    # split by video_id
    data_nonbow_train, data_nonbow_test = train_test_split_video_id(data, tt_split)

    print(data_nonbow_train)
    print(data_nonbow_test)

    assert set(data_nonbow_train[COL_VIDEO_ID]) == {'wwee', 'fsfs', 'mlem'}
    assert set(data_nonbow_test[COL_VIDEO_ID]) == {'asdf', 'blah'}
    assert data_nonbow_train['vals'].tolist() == [3, 4, 5, 6, 7, 8, 9, 14, 15, 16]
    assert data_nonbow_test['vals'].tolist() == [0, 1, 2, 10, 11, 12, 13]
    assert len(data_nonbow_train) == 10
    assert len(data_nonbow_test) == 7







if __name__ == '__main__':
    # test_resample_to_uniform_grid()
    test_train_test_split()