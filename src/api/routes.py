"""Routes for data stores"""

from typing import List, Tuple, Generator, Union, Optional
from pprint import pprint

import pandas as pd
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from db_engines.mysql_engine import MySQLEngine
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen
from ytpa_utils.sql_utils import make_sql_query
from ytpa_utils.val_utils import is_subset
from ytpa_utils.df_utils import df_dt_codec
from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (TIMESTAMP_CONVERSION_FMTS, WS_STREAM_TERM_MSG, WS_MAX_RECORDS_SEND,
                                           COL_TIMESTAMP_ACCESSED)
from src.etl.prefeaturization_etl_utils import etl_extract_tabular, get_etl_req_prefeats


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


router_root = APIRouter()
router_rawdata = APIRouter()
router_prefeatures = APIRouter()



""" Helper methods """
def get_mysql_engine_and_tablename(request: Request,
                                   key: str) \
        -> Tuple[MySQLEngine, str]:
    """Get MySQL engine and tablename"""
    engine = request.app.mysql_engine
    tablename = DB_VIDEOS_TABLES[key]
    return engine, tablename

def get_mongodb_engine(request: Request,
                       database: Optional[str] = None,
                       collection: Optional[str] = None) \
        -> MongoDBEngine:
    """Get MongoDB engine"""
    engine = request.app.mongodb_engine
    engine.set_db_info(database=database, collection=collection)
    return engine

def setup_rawdata_df_gen(etl_config_name: str, etl_config: dict) \
        -> Tuple[Generator[pd.DataFrame, None, None], dict, MySQLEngine]:
    """Setup DataFrame generator for MySQL queries."""
    etl_request = get_etl_req_prefeats(etl_config_name, etl_config)
    df_gen, info_tabular_extract, engine = etl_extract_tabular(etl_request)
    return df_gen, info_tabular_extract, engine

def setup_mongodb_df_gen(database: str,
                         collection: str,
                         extract_options: dict) \
        -> Generator[pd.DataFrame, None, None]:
    """Setup DataFrame generator for MongoDB queries."""
    df_gen = get_mongodb_records_gen(database,
                                     collection,
                                     DB_MONGO_CONFIG,
                                     filter=extract_options.get('filter'),
                                     projection=extract_options.get('projection'),
                                     distinct=extract_options.get('distinct'))
    return df_gen

async def gen_next_records(gen, websocket: WebSocket):
    """Get next record from a generator if possibe, otherwise shut down websocket."""
    try:
        return next(gen)
    except:
        # clean up and disconnect
        await websocket.send_json(WS_STREAM_TERM_MSG)
        raise WebSocketDisconnect

async def send_df_via_websocket(df: pd.DataFrame,
                                df_dt_encodings: dict,
                                websocket: WebSocket):
    """Send DataFrame in pieces over websocket"""
    i = 0
    while not (df_ := df.iloc[i * WS_MAX_RECORDS_SEND: (i + 1) * WS_MAX_RECORDS_SEND]).empty:
        if df_dt_encodings is not None:
            cnvs = {key: val for key, val in df_dt_encodings.items() if key in df_.columns}
            df_dt_codec(df_, cnvs, 'encode')  # make it JSONifiable
        data_send: List[dict] = df_.to_dict('records')
        await websocket.send_json(data_send)
        i += 1
    await websocket.send_json([])




""" Root """
@router_root.get("/", response_description="Test for liveness", response_model=str)
def root(request: Request):
    """Test for liveness"""
    return "Hi there. The YT Analytics API is available."



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
    df_gen, engine = None, None
    try:
        while True:
            # receive JSON data
            data_recv = await websocket.receive_json()

            # initialize DataFrame generator first time around
            if df_gen is None:
                assert is_subset(['name', 'extract'], data_recv)
                df_gen, _, engine = setup_rawdata_df_gen(data_recv['name'], data_recv)

            # check for next DataFrame and exit stream if finished
            df = await gen_next_records(df_gen, websocket)

            # send DataFrame in chunks
            await send_df_via_websocket(df, TIMESTAMP_CONVERSION_FMTS, websocket)
    except WebSocketDisconnect as e:
        del df_gen, engine
        print(e)




""" Prefeatures """
@router_prefeatures.get("/config", response_description="Get prefeaturization ETL configurations",
                        response_model=List[dict])
def get_prefeature_configs(request: Request):
    """Get all prefeaturization configs"""
    engine = get_mongodb_engine(request,
                                database=DB_FEATURES_NOSQL_DATABASE,
                                collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_prefeatures'])
    df = engine.find_many()
    recs = df.to_dict('records')
    return recs

@router_prefeatures.websocket("/data")
async def get_prefeatures(websocket: WebSocket):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    await websocket.accept()
    df_gen = None
    try:
        while True:
            # receive JSON data
            data_recv = await websocket.receive_json()

            # initialize DataFrame generator first time around
            if df_gen is None:
                assert is_subset(['extract'], data_recv)
                df_gen = setup_mongodb_df_gen(DB_FEATURES_NOSQL_DATABASE,
                                              DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
                                              data_recv['extract'])

            # check for next DataFrame and exit stream if finished
            df = await gen_next_records(df_gen, websocket)

            # send DataFrame in chunks
            await send_df_via_websocket(df, TIMESTAMP_CONVERSION_FMTS, websocket)
    except WebSocketDisconnect as e:
        del df_gen
        print(e)
