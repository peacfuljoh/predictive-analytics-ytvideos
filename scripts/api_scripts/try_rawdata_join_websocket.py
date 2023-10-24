
import asyncio
from datetime import datetime
import json
from typing import List

import websockets
import pandas as pd

from src.crawler.crawler.utils.misc_utils import df_dt_codec
from src.crawler.crawler.constants import (TIMESTAMP_CONVERSION_FMTS, WS_STREAM_TERM_MSG, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED)
from src.crawler.crawler.config import API_CONFIG


RAWDATA_JOIN_ENDPOINT = f"ws://{API_CONFIG['host']}:{API_CONFIG['port']}/rawdata/join"

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
                        df_dt_codec(df, TIMESTAMP_CONVERSION_FMTS, 'decode')
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
