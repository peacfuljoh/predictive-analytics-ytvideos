"""Script to experiment with client-side websocket functionality"""

import asyncio
import json
import requests
from pprint import pprint

from ytpa_api_utils.websocket_utils import run_dfs_stream_with_options, process_dfs_stream
from ytpa_api_utils.request_utils import get_configs_post

from src.crawler.crawler.constants import (COL_USERNAME, COL_TIMESTAMP_ACCESSED, PREFEATURES_ETL_CONFIG_COL,
                                           VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL)
from src.crawler.crawler.config import (DB_INFO, DB_MONGO_CONFIG, DB_MYSQL_CONFIG,
                                        RAWDATA_JOIN_ENDPOINT, CONFIGS_ENDPOINT, PREFEATURES_ENDPOINT,
                                        VOCABULARY_ENDPOINT, FEATURES_ENDPOINT)




stream_rawdata_join = True
stream_prefeatures = True
stream_vocabulary = True
stream_features = True




async def df_simple_processor(q_stream: asyncio.Queue):
    """Wrapper to run simple DataFrame processing"""
    process_dfs_stream_options = dict(print_df=False, print_count=True)
    await process_dfs_stream(q_stream, options=process_dfs_stream_options)





""" Main websocket stream functions """
# raw data "join"
if stream_rawdata_join:
    ETL_CONFIG_OPTIONS = {
        'name': 'test', # not used when loading raw data as in prefeatures ETL pipeline
        'extract': {
            'filters': {
                COL_TIMESTAMP_ACCESSED: [['2023-10-10 00:00:00.000', '2023-10-15 00:00:00.000']],
                COL_USERNAME: ["FoxNews", "NBCNews"]
            },
            'limit': 7217
        }
    }

    q_stream = asyncio.Queue()
    run_dfs_stream_with_options(RAWDATA_JOIN_ENDPOINT, ETL_CONFIG_OPTIONS, df_simple_processor, q_stream)

    print('\n' * 3)


# prefeatures
if stream_prefeatures:
    etl_config_prefeatures = 'test3'

    recs = get_configs_post(CONFIGS_ENDPOINT, 'prefeatures', config_name=etl_config_prefeatures)
    pprint(recs)

    ETL_CONFIG_OPTIONS = {
        'extract': {
            'filter': {
                COL_TIMESTAMP_ACCESSED: [['2023-10-15 00:00:00.000', '2023-10-16 00:00:00.000']],
                COL_USERNAME: ["FoxNews", "NBCNews"],
                PREFEATURES_ETL_CONFIG_COL: etl_config_prefeatures
            }
        }
    }

    q_stream = asyncio.Queue()
    run_dfs_stream_with_options(PREFEATURES_ENDPOINT, ETL_CONFIG_OPTIONS, df_simple_processor, q_stream)

    print('\n' * 3)


# vocabulary
if stream_vocabulary:
    etl_config_vocabulary = 'test21715'

    recs = get_configs_post(CONFIGS_ENDPOINT, 'vocabulary', config_name=etl_config_vocabulary)
    pprint(recs)

    ETL_CONFIG_OPTIONS = {
        VOCAB_ETL_CONFIG_COL: etl_config_vocabulary
        # VOCAB_TIMESTAMP_COL: '...'
    }

    response = requests.post(VOCABULARY_ENDPOINT, json=ETL_CONFIG_OPTIONS)
    recs = json.loads(response.text)

    print('\n' * 3)


# features
if stream_features:
    etl_config_name = 'test21715'

    recs = get_configs_post(CONFIGS_ENDPOINT, 'features', config_name=etl_config_name)
    pprint(recs)

    ETL_CONFIG_OPTIONS = {
        'extract': {
            'filter': {
                COL_TIMESTAMP_ACCESSED: [['2023-10-15 00:00:00.000', '2023-10-16 00:00:00.000']],
                COL_USERNAME: ["FoxNews", "NBCNews"],
                FEATURES_ETL_CONFIG_COL: etl_config_name
            }
        }
    }

    q_stream = asyncio.Queue()
    run_dfs_stream_with_options(FEATURES_ENDPOINT, ETL_CONFIG_OPTIONS, df_simple_processor, q_stream)

    print('\n' * 3)

