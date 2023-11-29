"""Routes for data stores"""

from typing import List, Tuple, Generator
from pprint import pprint

import pandas as pd
from fastapi import APIRouter, Request, WebSocket

from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import insert_records_from_dict, update_records_from_dict

from ytpa_utils.sql_utils import make_sql_query
from ytpa_utils.val_utils import is_subset, is_list_of_instances
from ytpa_utils.misc_utils import run_func_and_return_stdout

from ytpa_api_utils.websocket_utils import run_websocket_stream_server

from src.crawler.crawler.constants import (VOCAB_ETL_CONFIG_COL, VOCAB_TIMESTAMP_COL, COL_THUMBNAIL_URL, COL_VIDEO_ID,
                                           TIMESTAMP_CONVERSION_FMTS_ENCODE)
from src.api.routes_utils import (etl_load_vocab_from_db, get_mysql_engine_and_tablename,
                                  setup_rawdata_df_gen, setup_mongodb_df_gen, get_configs, get_mongodb_engine)
from src.api.app_secrets import (DB_INFO, DB_MONGO_CONFIG, DB_MYSQL_CONFIG, DB_VIDEOS_DATABASE, DB_VIDEOS_TABLES,
                                 DB_VIDEOS_NOSQL_DATABASE, DB_VIDEOS_NOSQL_COLLECTIONS, DB_FEATURES_NOSQL_COLLECTIONS,
                                 DB_FEATURES_NOSQL_DATABASE)
from src.etl.etl_request_utils import get_validated_etl_request
from src.crawler.crawler.utils.mongodb_utils_ytvideos import fetch_url_and_save_image





router_root = APIRouter()
router_rawdata = APIRouter()
router_prefeatures = APIRouter()
router_vocabulary = APIRouter()
router_config = APIRouter()
router_features = APIRouter()
router_models = APIRouter()
router_mongo = APIRouter()






""" Root """
@router_root.get("/", response_description="Test for liveness", response_model=str)
def root(request: Request):
    """Test for liveness"""
    return "Hi there. The YT Analytics API is available."



""" General-purpose MongoDB CRUD+ """
@router_mongo.post("/insert_one", response_description="Insert single record to MongoDB data store", response_model=str)
def insert_one_record(request: Request, data: dict):
    assert set(data) == {'database', 'collection', 'record'}
    assert isinstance(data['database'], str)
    assert isinstance(data['collection'], str)
    assert isinstance(data['record'], dict)

    def func():
        database = DB_INFO[data['database']]
        collection = DB_FEATURES_NOSQL_COLLECTIONS[data['collection']]
        engine = get_mongodb_engine(request, database=database, collection=collection)
        engine.insert_one(data['record'])

    return run_func_and_return_stdout(func)

@router_mongo.post("/insert_many", response_description="Insert multiple records to MongoDB data store", response_model=str)
def insert_many_records(request: Request, data: dict):
    # insert records
    assert set(data) == {'database', 'collection', 'records'}
    assert isinstance(data['database'], str)
    assert isinstance(data['collection'], str)
    assert is_list_of_instances(data['records'], dict)

    def func():
        database = DB_INFO[data['database']]
        collection = DB_FEATURES_NOSQL_COLLECTIONS[data['collection']]
        engine = get_mongodb_engine(request, database=database, collection=collection)
        print(f"insert_many_records() -> Inserting {len(data['records'])} records in collection {collection} "
              f"of database {database}")
        engine.insert_many(data['records'])

    return run_func_and_return_stdout(func)



""" Configs """
@router_config.post("/pull", response_description="Get config info", response_model=List[dict])
def get_configs_route(request: Request, opts: dict):
    return get_configs(request, opts.get('name'))




""" Raw data """
@router_rawdata.get("/users/pull", response_description="Get video usernames", response_model=List[str])
def get_video_usernames(request: Request):
    """Get all usernames"""
    engine, tablename = get_mysql_engine_and_tablename(request, 'users')
    query = make_sql_query(tablename)
    ids: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    ids: List[str] = [id_[0] for id_ in ids]
    return ids

@router_rawdata.post("/meta/pull", response_description="Get video metadata", response_model=List[tuple])
def get_video_metadata(request: Request, opts: dict):
    """Get video meta information"""
    engine, tablename = get_mysql_engine_and_tablename(request, 'meta')
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    return records

@router_rawdata.post("/meta/push", response_description="Inject video metadata", response_model=None)
def post_video_metadata(request: Request, data: dict):
    """Inject video meta data to database"""
    insert_records_from_dict(DB_VIDEOS_DATABASE,
                             DB_VIDEOS_TABLES['meta'],
                             data,
                             DB_MYSQL_CONFIG,
                             keys=list(data.keys()))

@router_rawdata.post("/stats/pull", response_description="Get video stats", response_model=List[tuple])
def get_video_stats(request: Request, opts: dict):
    """Get video meta information"""
    engine, tablename = get_mysql_engine_and_tablename(request, 'stats')
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    del engine
    return records

@router_rawdata.post("/stats/push", response_description="Inject video stats", response_model=None)
def post_video_stats(request: Request, data: dict):
    """Inject video stats information"""
    # update metadata and insert stats
    update_records_from_dict(DB_VIDEOS_DATABASE,
                             DB_VIDEOS_TABLES['meta'],
                             data,
                             DB_MYSQL_CONFIG,
                             another_condition='upload_date IS NULL') # to avoid overwriting timestamp_first_seen
    insert_records_from_dict(DB_VIDEOS_DATABASE,
                             DB_VIDEOS_TABLES['stats'],
                             data,
                             DB_MYSQL_CONFIG)

    # fetch and save thumbnail to MongoDB database
    if len(url := data[COL_THUMBNAIL_URL]) > 0:
        try:
            fetch_url_and_save_image(DB_VIDEOS_NOSQL_DATABASE,
                                     DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'],
                                     DB_MONGO_CONFIG,
                                     data[COL_VIDEO_ID],
                                     url,
                                     verbose=True)
        except:
            print(f'Exception during MongoDB database injection for {COL_THUMBNAIL_URL}.')

@router_rawdata.websocket("/join")
async def get_video_meta_stats_join(websocket: WebSocket):
    """Perform join query between meta and stats raw data tables and return a stream of records."""
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], MySQLEngine]:
        assert is_subset(['name', 'extract'], data_recv)
        data_recv['db'] = {
            'db_info': DB_INFO,
            'db_mysql_config': DB_MYSQL_CONFIG,
            'db_mongo_config': DB_MONGO_CONFIG
        }
        df_gen, _, engine = setup_rawdata_df_gen(data_recv['name'], data_recv)
        return df_gen, engine

    await websocket.accept()
    await run_websocket_stream_server(websocket, setup_df_gen, transformations=TIMESTAMP_CONVERSION_FMTS_ENCODE)




""" Prefeatures """
@router_prefeatures.websocket("/pull")
async def get_prefeatures(websocket: WebSocket):
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], None]:
        assert is_subset(['extract'], data_recv)
        df_gen = setup_mongodb_df_gen(DB_FEATURES_NOSQL_DATABASE,
                                      DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
                                      data_recv['extract'])
        return df_gen, None

    await websocket.accept()
    await run_websocket_stream_server(websocket, setup_df_gen)

def get_preconfig(request: Request,
                  collection: str,
                  etl_config_name: str) \
        -> dict:
    """Get preconfig corresponding to a specified config."""
    prefeatures_etl_configs = [cf for cf in get_configs(request, collection) if cf['_id'] == etl_config_name]
    assert len(prefeatures_etl_configs) == 1
    return prefeatures_etl_configs[0]['preconfig']




""" Vocabulary """
@router_vocabulary.post("/pull", response_description="Get vocabulary", response_model=dict)
def get_vocabulary(request: Request, opts: dict):
    # load vocabulary from db
    assert set(opts) == {VOCAB_ETL_CONFIG_COL, VOCAB_TIMESTAMP_COL}

    etl_config = {
        'preconfig': {
            **get_preconfig(request, 'vocabulary', opts[VOCAB_ETL_CONFIG_COL]),
            VOCAB_ETL_CONFIG_COL: opts[VOCAB_ETL_CONFIG_COL]
        },
        'db': {'db_info': DB_INFO, 'db_mongo_config': DB_MONGO_CONFIG}
    }
    req_features = get_validated_etl_request('features', etl_config)

    rec = etl_load_vocab_from_db(req_features,
                                 opts.get(VOCAB_TIMESTAMP_COL),
                                 as_gs_dict=False)
    rec['_id'] = str(rec['_id'])

    return rec




""" Features """
@router_features.websocket("/pull")
async def get_features(websocket: WebSocket):
    def setup_df_gen(data_recv: dict) -> Tuple[Generator[pd.DataFrame, None, None], None]:
        assert is_subset(['extract'], data_recv)
        df_gen = setup_mongodb_df_gen(DB_FEATURES_NOSQL_DATABASE,
                                      DB_FEATURES_NOSQL_COLLECTIONS['features'],
                                      data_recv['extract'])
        return df_gen, None

    await websocket.accept()
    await run_websocket_stream_server(websocket, setup_df_gen)




# """ Models """
# @router_models.post("/model", response_description="Get model", response_model=dict)
# def get_model(request: Request, opts: dict):
#     # load model from db
#     etl_config = {
#         'preconfig': {
#             **get_preconfig(request, 'vocabulary', opts[VOCAB_ETL_CONFIG_COL]),
#             VOCAB_ETL_CONFIG_COL: opts[VOCAB_ETL_CONFIG_COL]
#         },
#         'db': {'db_info': DB_INFO, 'db_mongo_config': DB_MONGO_CONFIG}
#     }
#     req_features = get_validated_etl_request('models', etl_config)
#
#     rec = etl_load_vocab_from_db(req_features,
#                                  opts.get(VOCAB_TIMESTAMP_COL),
#                                  as_gs_dict=False)
#     rec['_id'] = str(rec['_id'])
#
#     return rec
