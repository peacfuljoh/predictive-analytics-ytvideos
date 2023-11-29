"""Sketch of integration testing for API endpoints"""

import time
import requests

from ytpa_api_utils.websocket_utils import df_generator_ws
from src.crawler.crawler.config import (CONFIGS_ENDPOINT, MONGO_INSERT_ONE_ENDPOINT, MONGO_INSERT_MANY_ENDPOINT,
                                        RAWDATA_META_PULL_ENDPOINT, RAWDATA_META_PUSH_ENDPOINT,
                                        RAWDATA_STATS_PULL_ENDPOINT, RAWDATA_STATS_PUSH_ENDPOINT,
                                        RAWDATA_JOIN_ENDPOINT, PREFEATURES_ENDPOINT, VOCABULARY_ENDPOINT,
                                        FEATURES_ENDPOINT)
from src.crawler.crawler.constants import COL_USERNAME, TIMESTAMP_CONVERSION_FMTS_DECODE



DUR = 1 * 60.0 # seconds


def endpoint_caller(func):
    def inner(*args, **kwargs):
        t_start = time.time()
        num_calls = 0
        while time.time() - t_start < DUR:
            func(*args, **kwargs)
            num_calls += 1
        print(f'num_calls: {num_calls}')
    return inner



@endpoint_caller
def call_configs_endpoint():
    for name_ in ['prefeatures', 'vocabulary', 'features']:
        config = requests.post(CONFIGS_ENDPOINT, json={'name': name_})

@endpoint_caller
def call_rawdata_meta_pull():
    opts = dict(
        cols=None,
        where=None,
        limit=1000
    )
    requests.post(RAWDATA_META_PULL_ENDPOINT, json=opts)

@endpoint_caller
def call_rawdata_stats_pull():
    opts = dict(
        cols=None,
        where=None,
        limit=1000
    )
    requests.post(RAWDATA_STATS_PULL_ENDPOINT, json=opts)

@endpoint_caller
def call_rawdata_join_websocket():
    etl_config_name = 'test3'
    extract_options = {
        'filters': {
            COL_USERNAME: ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
        }
    }

    etl_config_options = {'name': etl_config_name, 'extract': extract_options}
    df_gen = df_generator_ws(RAWDATA_JOIN_ENDPOINT, etl_config_options,
                             transformations=TIMESTAMP_CONVERSION_FMTS_DECODE)
    for _ in df_gen:
        pass





if __name__ == '__main__':
    # call_configs_endpoint()
    # call_rawdata_meta_pull()
    # call_rawdata_stats_pull()
    call_rawdata_join_websocket()

