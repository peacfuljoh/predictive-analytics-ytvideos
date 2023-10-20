"""Routes for data stores"""

from typing import List, Tuple, Generator
from pprint import pprint

import pandas as pd
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from src.crawler.crawler.utils.mysql_engine import MySQLEngine
from src.crawler.crawler.utils.misc_utils import make_sql_query, is_subset, df_dt_codec
from src.crawler.crawler.config import DB_INFO
from src.crawler.crawler.constants import TIMESTAMP_CONVERSION_FMTS, WS_STREAM_TERM_MSG, WS_MAX_RECORDS_SEND
from src.etl.prefeaturization_etl_utils import etl_extract_tabular, get_etl_req_prefeats


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']

router_root = APIRouter()
router_rawdata = APIRouter()



""" Setup """
def get_mysql_engine_and_tablename(request: Request,
                                   key: str) \
        -> Tuple[MySQLEngine, str]:
    """Get MySQL engine and tablename"""
    engine = request.app.mysql_engine
    tablename = DB_VIDEOS_TABLES[key]

    return engine, tablename



""" Root """
@router_root.get("/", response_description="Test for liveness", response_model=str)
def get_usernames(request: Request):
    """Test for liveness"""
    return "The app is running"



""" Raw data """
@router_rawdata.get("/users", response_description="Get video usernames", response_model=List[str])
def get_video_usernames(request: Request):
    """Get all usernames"""
    engine, tablename = get_mysql_engine_and_tablename(request, 'users')
    query = make_sql_query(tablename)
    ids: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    ids: List[str] = [id_[0] for id_ in ids]
    return ids

@router_rawdata.post("/meta", response_description="Get video metadata", response_model=List[tuple])
def get_video_metadata(request: Request, opts: dict):
    """Get video meta information"""
    pprint(opts)
    engine, tablename = get_mysql_engine_and_tablename(request, 'meta')
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    return records

@router_rawdata.post("/stats", response_description="Get video stats", response_model=List[tuple])
def get_video_stats(request: Request, opts: dict):
    """Get video meta information"""
    pprint(opts)
    engine, tablename = get_mysql_engine_and_tablename(request, 'stats')
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    return records

@router_rawdata.websocket("/join")
async def get_video_meta_stats_join(websocket: WebSocket):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    await websocket.accept()
    try:
        df_gen = None
        info_tabular_extract = None
        engine = None

        def setup_df_gen(etl_config_name: str, extract_options: dict) \
                -> Tuple[Generator[pd.DataFrame, None, None], dict, MySQLEngine]:
            """Setup for DataFrame generator."""
            etl_config = dict(extract=extract_options)
            etl_request = get_etl_req_prefeats(etl_config_name, etl_config)
            df_gen, info_tabular_extract, engine = etl_extract_tabular(etl_request)
            return df_gen, info_tabular_extract, engine

        while True:
            # receive JSON data
            data_recv = await websocket.receive_json()

            # initialize DataFrame generator
            if df_gen is None:
                assert is_subset(['name', 'extract'], data_recv)
                df_gen, info_tabular_extract, engine = setup_df_gen(data_recv['name'], data_recv['extract'])

            # check for next DataFrame and exit stream if finished
            df = next(df_gen)
            if df.empty: # clean up and disconnect
                del engine
                await websocket.send_json(WS_STREAM_TERM_MSG)
                raise WebSocketDisconnect

            # make it JSONifiable
            df_dt_codec(df, TIMESTAMP_CONVERSION_FMTS, 'encode')

            # send DataFrame in chunks
            i = 0
            while not (df_ := df.iloc[i * WS_MAX_RECORDS_SEND: (i + 1) * WS_MAX_RECORDS_SEND]).empty:
                data_send: List[dict] = df_.to_dict('records')
                await websocket.send_json(data_send)
                i += 1
            await websocket.send_json([])

    except WebSocketDisconnect as e:
        print(e)
        # manager.disconnect(websocket)
        # await manager.broadcast(f"Client #{client_id} left the chat")
