
import asyncio
import json
from typing import List, Union, Optional
import requests
from pprint import pprint
import queue

import websockets
import pandas as pd

from src.crawler.crawler.constants import (WS_STREAM_TERM_MSG, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, PREFEATURES_ETL_CONFIG_COL,
                                           VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL)
from src.crawler.crawler.config import API_CONFIG, DB_INFO, DB_MONGO_CONFIG, DB_MYSQL_CONFIG


RAWDATA_JOIN_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/rawdata/join"
CONFIGS_ENDPOINT = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/config"
PREFEATURES_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/prefeatures/data"
VOCABULARY_ENDPOINT = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/vocabulary/data"
FEATURES_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/features/data"



stream_rawdata_join = False
stream_prefeatures = False
stream_vocabulary = False
stream_features = True




""" Websocket data stream handling methods """
async def get_next_msg(websocket: websockets.connect) -> Optional[List[dict]]:
    """Receive next message over websocket"""
    data_recv = await websocket.recv()
    data_recv: Union[List[dict], str] = json.loads(data_recv)
    if data_recv == WS_STREAM_TERM_MSG:
        raise StopIteration
    if len(data_recv) == 0:
        return
    return data_recv

async def receive_msgs(websocket: websockets.connect,
                       q: asyncio.Queue):
    """Receive sequence of data messages over websocket connection, placing them in a queue."""
    while 1:
        data_recv = await get_next_msg(websocket)
        if data_recv is None:
            return
        df = pd.DataFrame.from_dict(data_recv)
        await q.put(df)

async def stream_dfs_websocket(endpoint: str,
                               options: dict,
                               q: asyncio.Queue):
    """Stream data into a queue of DataFrames over a websocket."""
    options_str = json.dumps(options)
    async with websockets.connect(endpoint) as websocket:
        try:
            while 1:
                await websocket.send(options_str)
                await receive_msgs(websocket, q)
        except Exception as e:
            print(e)
            await q.put(None)

async def process_dfs_stream(q_stream: asyncio.Queue,
                             options: Optional[dict] = None):
    """Print DataFrames from a queue until a None is found"""
    # options
    if options is None:
        options = {}
    print_dfs = options.get('print_df')
    print_count = options.get('print_count')
    q_stats = options.get('q_stats')

    # process stream
    count = 0
    while 1:
        df = await q_stream.get()
        if df is None:
            return
        count += len(df)
        if print_dfs:
            print(df)
        if print_count:
            print(f"Records count so far: {count}.")
        if q_stats:
            q_stats.put({'count': count})

def run_dfs_stream_with_options(endpoint: str,
                                etl_config: dict,
                                q_stream: asyncio.Queue,
                                process_options: dict):
    """Simultaneously stream data through websocket and process it."""
    async def run_tasks():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(process_dfs_stream(q_stream, process_options))
            tg.create_task(stream_dfs_websocket(endpoint, etl_config, q_stream))
    asyncio.run(run_tasks())







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
        },
        'db': {
            'db_info': DB_INFO,
            'db_mysql_config': DB_MYSQL_CONFIG,
            'db_mongo_config': DB_MONGO_CONFIG
        }
    }

    q_stream = asyncio.Queue()
    process_dfs_stream_options = dict(print_df=True, print_count=True)
    run_dfs_stream_with_options(RAWDATA_JOIN_ENDPOINT, ETL_CONFIG_OPTIONS, q_stream, process_dfs_stream_options)

    print('\n' * 5)



# prefeatures
if stream_prefeatures:
    # configs
    response = requests.post(CONFIGS_ENDPOINT, json=dict(name='prefeatures'))
    recs = json.loads(response.text)
    pprint(recs)

    # data
    etl_config_prefeatures = 'test3'

    config_ids = [config_['_id'] for config_ in recs]
    assert etl_config_prefeatures in config_ids

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
    process_dfs_stream_options = dict(print_df=False, print_count=True)
    run_dfs_stream_with_options(PREFEATURES_ENDPOINT, ETL_CONFIG_OPTIONS, q_stream, process_dfs_stream_options)

    print('\n' * 5)


# vocabulary
if stream_vocabulary:
    # configs
    response = requests.post(CONFIGS_ENDPOINT, json=dict(name='vocabulary'))
    recs = json.loads(response.text)
    pprint(recs)

    # data
    etl_config_vocabulary = 'test21715'

    config_ids = [config_['_id'] for config_ in recs]
    assert etl_config_vocabulary in config_ids

    ETL_CONFIG_OPTIONS = {
        VOCAB_ETL_CONFIG_COL: etl_config_vocabulary
        # VOCAB_TIMESTAMP_COL: '...'
    }

    response = requests.post(VOCABULARY_ENDPOINT, json=ETL_CONFIG_OPTIONS)
    recs = json.loads(response.text)

    print('\n' * 5)


# features
if stream_features:
    # configs
    response = requests.post(CONFIGS_ENDPOINT, json=dict(name='features'))
    recs = json.loads(response.text)
    pprint(recs)

    # data
    etl_config_name = 'test21715'

    config_ids = [config_['_id'] for config_ in recs]
    assert etl_config_name in config_ids

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
    process_dfs_stream_options = dict(print_df=False, print_count=True)
    run_dfs_stream_with_options(FEATURES_ENDPOINT, ETL_CONFIG_OPTIONS, q_stream, process_dfs_stream_options)

    print('\n' * 5)

