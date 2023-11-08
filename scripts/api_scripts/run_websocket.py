
import asyncio
from datetime import datetime
import json
from typing import List
import requests
from pprint import pprint

import websockets
import pandas as pd

from ytpa_utils.df_utils import df_dt_codec
from src.crawler.crawler.constants import (TIMESTAMP_CONVERSION_FMTS, WS_STREAM_TERM_MSG, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, PREFEATURES_ETL_CONFIG_COL)
from src.crawler.crawler.config import API_CONFIG, DB_INFO, DB_MONGO_CONFIG, DB_MYSQL_CONFIG


RAWDATA_JOIN_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/rawdata/join"
PREFEATURE_CONFIGS_ENDPOINT = f"http://{API_CONFIG['host']}:{API_CONFIG['port']}/prefeatures/config"
PREFEATURES_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/prefeatures/data"


# raw data "join"
if 0:
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
    ETL_CONFIG_OPTIONS_STR = json.dumps(ETL_CONFIG_OPTIONS)


    async def stream_meta_stats_join():
        count = 0
        async with websockets.connect(RAWDATA_JOIN_ENDPOINT) as websocket:
            while 1:
                try:
                    # msg = dict(msg=f"The time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.")
                    await websocket.send(ETL_CONFIG_OPTIONS_STR)

                    while 1:
                        data_recv = await websocket.recv()
                        data_recv: List[dict] = json.loads(data_recv)
                        if data_recv == WS_STREAM_TERM_MSG:
                            return
                        if len(data_recv) > 0:
                            df = pd.DataFrame.from_dict(data_recv)
                            count += len(df)
                            # df_dt_codec(df, TIMESTAMP_CONVERSION_FMTS, 'decode') # decoding not implemented in df_dt_codec
                            # print(df[COL_TIMESTAMP_ACCESSED].iloc[0])
                            # print(df[COL_UPLOAD_DATE].iloc[0])
                        else:
                            break

                    print(f'Total records received so far: {count}.')
                    # await asyncio.sleep(1)
                except Exception as e:
                    print(e)
                    break


    # asyncio.get_event_loop().run_until_complete(stream_meta_stats_join())
    asyncio.run(stream_meta_stats_join())

# prefeatures and config
if 1:
    response = requests.get(PREFEATURE_CONFIGS_ENDPOINT)
    recs = json.loads(response.text)
    pprint(recs)

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
    ETL_CONFIG_OPTIONS_STR = json.dumps(ETL_CONFIG_OPTIONS)


    async def stream_prefeatures():
        count = 0
        async with websockets.connect(PREFEATURES_ENDPOINT) as websocket:
            while 1:
                try:
                    # msg = dict(msg=f"The time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.")
                    await websocket.send(ETL_CONFIG_OPTIONS_STR)

                    while 1:
                        data_recv = await websocket.recv()
                        data_recv: List[dict] = json.loads(data_recv)
                        if data_recv == WS_STREAM_TERM_MSG:
                            return
                        if len(data_recv) > 0:
                            df = pd.DataFrame.from_dict(data_recv)
                            count += len(df)
                            print(df[COL_TIMESTAMP_ACCESSED])
                            print(df[PREFEATURES_ETL_CONFIG_COL])
                            pprint(df)
                        else:
                            break

                    print(f'Total records received so far: {count}.')
                    # await asyncio.sleep(1)
                except Exception as e:
                    print(e)
                    break


    # asyncio.get_event_loop().run_until_complete(stream_meta_stats_join())
    asyncio.run(stream_prefeatures())


