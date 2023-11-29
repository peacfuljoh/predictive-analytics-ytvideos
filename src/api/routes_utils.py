"""Util methods for API endpoints"""

from typing import List, Tuple, Generator, Optional

import pandas as pd
from fastapi import Request

from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import perform_join_mysql_query
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen
from ytpa_utils.gensim_utils import convert_string_to_gs_dictionary

from src.crawler.crawler.constants import (VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL, VOCAB_ETL_CONFIG_COL,
                                           STATS_ALL_COLS, META_ALL_COLS_NO_URL, COL_VIDEO_ID)
from src.etl.etl_request_utils import get_validated_etl_request
from src.etl.etl_request import ETLRequestPrefeatures, ETLRequestFeatures
from src.api.app_secrets import (DB_MONGO_CONFIG, DB_MYSQL_CONFIG, DB_VIDEOS_TABLES, DB_FEATURES_NOSQL_COLLECTIONS,
                                 DB_FEATURES_NOSQL_DATABASE)






""" Endpoint helper methods """
def get_mysql_engine_and_tablename(request: Request,
                                   key: str) \
        -> Tuple[MySQLEngine, str]:
    """Get MySQL engine and tablename"""
    # engine = request.app.mysql_engine
    engine = MySQLEngine(DB_MYSQL_CONFIG)
    tablename = DB_VIDEOS_TABLES[key]
    return engine, tablename

def get_mongodb_engine(request: Request,
                       database: Optional[str] = None,
                       collection: Optional[str] = None) \
        -> MongoDBEngine:
    """Get MongoDB engine"""
    # engine = request.app.mongodb_engine
    engine = MongoDBEngine(DB_MONGO_CONFIG, verbose=True)
    engine.set_db_info(database=database, collection=collection)
    return engine

def setup_rawdata_df_gen(etl_config_name: str,
                         etl_config: dict) \
        -> Tuple[Generator[pd.DataFrame, None, None], dict, MySQLEngine]:
    """Setup DataFrame generator for MySQL queries."""
    etl_request = get_validated_etl_request('prefeatures', etl_config, etl_config_name)
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

def get_mongodb_records(request, database: str, collection: str) -> List[dict]:
    """Get records from MongoDB record store."""
    engine = get_mongodb_engine(request, database=database, collection=collection)
    df = engine.find_many()
    recs = df.to_dict('records')
    return recs

def get_configs(request: Request,
                collection: str) \
        -> List[dict]:
    """Get list of configs for various pipelines."""
    if collection == 'prefeatures':
        return get_mongodb_records(request,
                                   DB_FEATURES_NOSQL_DATABASE,
                                   DB_FEATURES_NOSQL_COLLECTIONS['etl_config_prefeatures'])
    if collection == 'vocabulary':
        return get_mongodb_records(request,
                                   DB_FEATURES_NOSQL_DATABASE,
                                   DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'])
    if collection == 'features':
        return get_mongodb_records(request,
                                   DB_FEATURES_NOSQL_DATABASE,
                                   DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'])
    return []






""" Prefeatures """
def etl_extract_tabular(req: ETLRequestPrefeatures) -> Tuple[Generator[pd.DataFrame, None, None], dict, MySQLEngine]:
    """Extract tabular raw_data according to request"""
    db_ = req.get_db()

    # required inputs
    database = db_['db_info']['DB_VIDEOS_DATABASE']
    tablename_primary = db_['db_info']['DB_VIDEOS_TABLES']['stats']
    tablename_secondary = db_['db_info']['DB_VIDEOS_TABLES']['meta']
    table_pseudoname_primary = 'stats'
    table_pseudoname_secondary = 'meta'
    join_condition = f'{table_pseudoname_primary}.video_id = {table_pseudoname_secondary}.video_id'
    cols_all = {
        table_pseudoname_primary: STATS_ALL_COLS,
        table_pseudoname_secondary: [col for col in META_ALL_COLS_NO_URL if col != COL_VIDEO_ID]
    }
    extract_ = req.get_extract()
    filters = extract_.get('filters')
    limit = extract_.get('limit')

    # perform query
    df, engine = perform_join_mysql_query(
        db_['db_mysql_config'],
        database,
        tablename_primary,
        tablename_secondary,
        table_pseudoname_primary,
        table_pseudoname_secondary,
        join_condition,
        cols_all,
        filters=filters,
        limit=limit,
        as_generator=True
    )

    # collect info from extraction
    extract_info = dict(
        table_pseudoname_primary=table_pseudoname_primary,
        table_pseudoname_secondary=table_pseudoname_secondary
    )

    return df, extract_info, engine




""" Vocabulary """
def etl_load_vocab_meta_from_db(engine: MongoDBEngine,
                                filter: Optional[dict] = None) \
        -> pd.DataFrame:
    """Load vocabulary info without vocab itself. Used to inspect metadata."""
    return engine.find_many(filter=filter, projection={VOCAB_VOCABULARY_COL: 0})

def etl_load_vocab_setup_engine(req: ETLRequestFeatures):
    """Setup MongoDB engine for loading vocabulary records."""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_vocab = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['vocabulary']

    return MongoDBEngine(mongo_config, database=database, collection=collection_vocab, verbose=True)

def etl_load_vocab_from_db(req: ETLRequestFeatures,
                           timestamp_vocab: Optional[str] = None,
                           as_gs_dict: bool = True) \
        -> dict:
    """Load vocabulary given specified options"""
    engine = etl_load_vocab_setup_engine(req)

    # get records (choose timestamp before loading entire vocabulary)
    etl_config_name = req.get_preconfig()[VOCAB_ETL_CONFIG_COL]
    filter = {VOCAB_ETL_CONFIG_COL: etl_config_name}
    filter[VOCAB_TIMESTAMP_COL] = timestamp_vocab if timestamp_vocab is not None else \
        etl_load_vocab_meta_from_db(engine, filter=filter)[VOCAB_TIMESTAMP_COL].max()
    df = engine.find_many(filter=filter)
    assert len(df) == 1
    rec: dict = df.iloc[0].to_dict()

    # convert vocab string to Dictionary object
    assert VOCAB_VOCABULARY_COL in rec
    if as_gs_dict:
        rec[VOCAB_VOCABULARY_COL] = convert_string_to_gs_dictionary(rec[VOCAB_VOCABULARY_COL])

    return rec


