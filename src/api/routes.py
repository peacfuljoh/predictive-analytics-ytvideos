"""Routes for data stores"""

from typing import List, Tuple, Generator, Union, Optional, Callable
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
                                           VOCAB_ETL_CONFIG_COL, ETL_CONFIG_VALID_KEYS_FEATURES,
                                           ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
from src.etl.prefeaturization_etl_utils import etl_extract_tabular, get_etl_req_prefeats
from src.etl.featurization_etl_utils import etl_load_vocab_from_db, ETLRequestFeatures


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


router_root = APIRouter()
router_rawdata = APIRouter()
router_prefeatures = APIRouter()
router_vocabulary = APIRouter()
router_config = APIRouter()
router_features = APIRouter()



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

def get_mongodb_records(request, database: str, collection: str) -> List[dict]:
    """Get records from MongoDB record store."""
    engine = get_mongodb_engine(request, database=database, collection=collection)
    df = engine.find_many()
    recs = df.to_dict('records')
    return recs

async def run_websocket_stream(websocket: WebSocket,
                               setup_df_gen: Callable):
    """Run websocket stream that generates DataFrames."""
    df_gen, engine = None, None
    try:
        while True:
            data_recv = await websocket.receive_json() # receive JSON data
            if df_gen is None: # initialize DataFrame generator first time around
                df_gen, engine = setup_df_gen(data_recv)
            df = await gen_next_records(df_gen, websocket)
            await send_df_via_websocket(df, TIMESTAMP_CONVERSION_FMTS, websocket) # send DataFrame in chunks
    except WebSocketDisconnect as e:
        del df_gen, engine
        print(e)






""" Root """
@router_root.get("/", response_description="Test for liveness", response_model=str)
def root(request: Request):
    """Test for liveness"""
    return "Hi there. The YT Analytics API is available."



""" Configs """
@router_config.post("/", response_description="Get config info", response_model=List[dict])
def get_configs(request: Request, opts: dict):
    config_name = opts.get('name')
    if config_name == 'prefeatures':
        return get_mongodb_records(request,
                                   DB_FEATURES_NOSQL_DATABASE,
                                   DB_FEATURES_NOSQL_COLLECTIONS['etl_config_prefeatures'])
    if config_name == 'vocabulary':
        return get_mongodb_records(request,
                                   DB_FEATURES_NOSQL_DATABASE,
                                   DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'])
    return []



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
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], MySQLEngine]:
        assert is_subset(['name', 'extract'], data_recv)
        df_gen, _, engine = setup_rawdata_df_gen(data_recv['name'], data_recv)
        return df_gen, engine

    await websocket.accept()
    await run_websocket_stream(websocket, setup_df_gen)




""" Prefeatures """
@router_prefeatures.websocket("/")
async def get_prefeatures(websocket: WebSocket):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], None]:
        assert is_subset(['extract'], data_recv)
        df_gen = setup_mongodb_df_gen(DB_FEATURES_NOSQL_DATABASE,
                                      DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
                                      data_recv['extract'])
        return df_gen, None

    await websocket.accept()
    await run_websocket_stream(websocket, setup_df_gen)




""" Vocabulary """
@router_vocabulary.post("/", response_description="Get vocabulary", response_model=dict)
def get_vocabulary(request: Request, opts: dict):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    etl_config_name = opts['etl_config_name']

    etl_config = {
        'preconfig': {VOCAB_ETL_CONFIG_COL: etl_config_name},
        'db': {'db_info': DB_INFO, 'db_mongo_config': DB_MONGO_CONFIG}
    }

    req_features = ETLRequestFeatures(etl_config,
                                      etl_config_name,
                                      ETL_CONFIG_VALID_KEYS_FEATURES,
                                      ETL_CONFIG_EXCLUDE_KEYS_FEATURES)

    return etl_load_vocab_from_db(req_features,
                                  opts.get('timestamp_vocab'),
                                  as_gs_dict=False)




""" Features """
@router_features.websocket("/")
async def get_features(websocket: WebSocket):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], None]:
        assert is_subset(['extract'], data_recv)
        df_gen = setup_mongodb_df_gen(DB_FEATURES_NOSQL_DATABASE,
                                      DB_FEATURES_NOSQL_COLLECTIONS['features'],
                                      data_recv['extract'])
        return df_gen, None

    await websocket.accept()
    await run_websocket_stream(websocket, setup_df_gen)

