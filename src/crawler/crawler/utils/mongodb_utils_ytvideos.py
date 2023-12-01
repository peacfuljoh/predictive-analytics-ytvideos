"""Utils for interfacing with the MongoDB database"""
from typing import Optional

import pandas as pd

from ytpa_utils.misc_utils import fetch_data_at_url
from ytpa_utils.val_utils import is_subset
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen, load_all_recs_with_distinct
from ..constants import (VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL, PREFEATURES_ETL_CONFIG_COL,
                         VOCAB_TIMESTAMP_COL, FEATURES_TIMESTAMP_COL)
from ..config import DB_INFO, DB_MONGO_CONFIG


def save_image_to_db(database: str,
                     collection: str,
                     db_config: dict,
                     video_id: str,
                     image_data: bytes,
                     verbose: bool = False):
    """Insert image to MongoDB collection"""
    engine = MongoDBEngine(db_config, database=database, collection=collection, verbose=verbose)
    record = {'_id': video_id, 'img': image_data}
    engine.insert_one(record)

def fetch_url_and_save_image(database: str,
                             collection: str,
                             db_config: dict,
                             video_id: str,
                             url: str,
                             delay: int = 0,
                             verbose: bool = False):
    """Fetch image and insert into MongoDB collection, but only if it isn't already there."""
    engine = MongoDBEngine(db_config, database=database, collection=collection, verbose=verbose)
    if engine.find_one_by_id(id=video_id) is None:
        record = {'_id': video_id, 'img': fetch_data_at_url(url, delay=delay)}
        engine.insert_one(record)


def load_config_timestamp_sets_for_features(configs: Optional[dict] = None) -> pd.DataFrame:
    """
    Load timestamped config info for features collection.
    """
    # setup
    database = DB_INFO['DB_FEATURES_NOSQL_DATABASE']
    collections = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']
    collection = collections['features']
    db_config = DB_MONGO_CONFIG

    cols_all = [PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL, FEATURES_ETL_CONFIG_COL,
                VOCAB_TIMESTAMP_COL, FEATURES_TIMESTAMP_COL]

    # ensure specified cols are valid
    if configs is not None:
        assert is_subset(configs, cols_all)
        cols_all = [col for col in cols_all if col not in configs] # remove specified names from search, maintain order

    # if unique record specified, find and return it
    if len(cols_all) == 0:
        df_gen = get_mongodb_records_gen(database, collection, db_config, filter=configs)
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
                df = load_all_recs_with_distinct(database, collection, db_config, group)
            else:
                df = load_all_recs_with_distinct(database, collection, db_config, group, filter={'$match': configs})
            continue

        # iterate over unique combos so far
        dfs = []
        for _, rec in df.iterrows():
            filter = {'$match': rec.to_dict()}  # limit search to this combo
            if configs is not None:
                filter['$match'] = {**filter['$match'], **configs}
            df_ = load_all_recs_with_distinct(database, collection, db_config, group, filter=filter)
            rec_tiled = pd.concat([rec.to_frame().T] * len(df_), axis=0, ignore_index=True) # tile rows to size of df_
            df_ = pd.concat((rec_tiled, df_), axis=1)
            dfs.append(df_)
        df = pd.concat(dfs, axis=0, ignore_index=True)

    # add back any pre-specified config options
    if configs is not None:
        for key, val in configs.items():
            df[key] = val

    return df


