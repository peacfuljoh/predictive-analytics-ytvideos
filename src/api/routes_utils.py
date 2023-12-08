"""Util methods for API endpoints"""

from typing import List, Tuple, Generator, Optional

import pandas as pd
from fastapi import Request

from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import perform_join_mysql_query
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen, load_all_recs_with_distinct
from ytpa_utils.gensim_utils import convert_string_to_gs_dictionary

from src.crawler.crawler.constants import (VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL, VOCAB_ETL_CONFIG_COL,
                                           STATS_ALL_COLS, META_ALL_COLS_NO_URL, COL_VIDEO_ID,
                                           FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL)
from src.etl.etl_request_utils import get_validated_etl_request
from src.etl.etl_request import ETLRequest, ETLRequestPrefeatures, ETLRequestFeatures, req_to_etl_config_record
from src.api.app_secrets import (DB_MONGO_CONFIG, DB_MYSQL_CONFIG, DB_VIDEOS_TABLES, DB_FEATURES_NOSQL_COLLECTIONS,
                                 DB_FEATURES_NOSQL_DATABASE)
from ytpa_utils.val_utils import is_subset




""" Endpoint helper methods """
def get_mysql_engine_and_tablename(key: str) \
        -> Tuple[MySQLEngine, str]:
    """Get MySQL engine and tablename"""
    engine = MySQLEngine(DB_MYSQL_CONFIG)
    tablename = DB_VIDEOS_TABLES[key]
    return engine, tablename

def get_mongodb_engine(database: Optional[str] = None,
                       collection: Optional[str] = None) \
        -> MongoDBEngine:
    """Get MongoDB engine"""
    engine = MongoDBEngine(DB_MONGO_CONFIG, verbose=True)
    engine.set_db_info(database=database, collection=collection)
    return engine

def setup_rawdata_df_gen(etl_config_name: str,
                         etl_config: dict) \
        -> Tuple[Generator[pd.DataFrame, None, None], dict, MySQLEngine]:
    """Setup DataFrame generator for MySQL queries."""
    etl_request = get_validated_etl_request('prefeatures', etl_config, etl_config_name,
                                            validation_func=validation_func_local)
    df_gen, info_tabular_extract, engine = etl_extract_tabular(etl_request)
    return df_gen, info_tabular_extract, engine

def validation_func_local(req: ETLRequest,
                          collection: str):
    """Validation function local to API that avoids calling config validation endpoint (freezes inside a websocket)."""
    config = req_to_etl_config_record(req, 'subset')
    is_valid = validate_config(config, collection)
    if not is_valid:
        raise Exception(f'The specified ETL pipeline options do not match those of '
                        f'the existing config for name {req.name}.')
    req.set_valid(True)

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
    engine = get_mongodb_engine(database=database, collection=collection)
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

def validate_config(config: dict,
                    collection: str) \
        -> bool:
    """Validate provided config against existing configs in a specified collection"""
    database = DB_FEATURES_NOSQL_DATABASE
    collection = DB_FEATURES_NOSQL_COLLECTIONS['etl_config_' + collection]

    engine = get_mongodb_engine(database=database, collection=collection)
    config_exist = engine.find_one_by_id(config['_id'])  # existing config

    # ensure that configs with same id match exactly
    if (config_exist is not None) and (config != config_exist):
        return False
    return True

def get_preconfig(request: Request,
                  collection: str,
                  etl_config_name: str) \
        -> dict:
    """Get preconfig corresponding to a specified config."""
    prefeatures_etl_configs = [cf for cf in get_configs(request, collection) if cf['_id'] == etl_config_name]
    assert len(prefeatures_etl_configs) == 1
    return prefeatures_etl_configs[0]['preconfig']





""" Configs """
def load_config_timestamp_sets_for_features(configs: Optional[dict] = None) -> pd.DataFrame:
    """
    Load timestamped config info for features collection.
    """
    # setup
    database = DB_FEATURES_NOSQL_DATABASE
    collection = DB_FEATURES_NOSQL_COLLECTIONS['features']

    cols_all = [PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                VOCAB_TIMESTAMP_COL, FEATURES_TIMESTAMP_COL]

    # ensure specified cols are valid
    if configs is not None:
        assert is_subset(configs, cols_all)
        cols_all = [col for col in cols_all if col not in configs] # remove specified names from search, maintain order

    # if unique record specified, find and return it
    if len(cols_all) == 0:
        df_gen = get_mongodb_records_gen(database, collection, DB_MONGO_CONFIG, filter=configs)
        dfs = [df for df in df_gen]
        assert len(dfs) == 1
        return dfs[0]

    # breadth-first search on config/timestamp combinations
    # TODO: find way to query unique sets at once instead of iterating in a worst-case-exponential fashion
    df = pd.DataFrame()
    for i, group in enumerate(cols_all):
        # first col name
        if i == 0:
            if configs is None:
                df = load_all_recs_with_distinct(database, collection, DB_MONGO_CONFIG, group)
            else:
                df = load_all_recs_with_distinct(database, collection, DB_MONGO_CONFIG, group,
                                                 filter={'$match': configs})
            continue

        # iterate over unique combos so far
        dfs = []
        for _, rec in df.iterrows():
            filter = {'$match': rec.to_dict()}  # limit search to this combo
            if configs is not None:
                filter['$match'] = {**filter['$match'], **configs}
            df_ = load_all_recs_with_distinct(database, collection, DB_MONGO_CONFIG, group, filter=filter)
            rec_tiled = pd.concat([rec.to_frame().T] * len(df_), axis=0, ignore_index=True) # tile rows to size of df_
            df_ = pd.concat((rec_tiled, df_), axis=1)
            dfs.append(df_)
        df = pd.concat(dfs, axis=0, ignore_index=True)

    # add back any pre-specified config options
    if configs is not None:
        for key, val in configs.items():
            df[key] = val

    return df



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

