"""
Preprocessor functionality:
- ETL
    - extract: load from databases
    - transform:
        - clean up raw data, apply filters
        - handle missing fields (impute, mark, invalidate)
        - featurize
    - load: send features to feature store
"""

from typing import Any, List, Dict, Union, Tuple
import datetime
from PIL import Image

import pandas as pd
import numpy as np

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.utils.db_mongo_utils import MongoDBEngine, get_mongodb_records
from src.crawler.crawler.utils.misc_utils import convert_bytes_to_image


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']
DB_NOSQL_DATABASE = DB_INFO['DB_NOSQL_DATABASE']
DB_NOSQL_COLLECTION_NAMES = DB_INFO['DB_NOSQL_COLLECTION_NAMES']


def is_datetime_formatted_str(s: Any, fmt: str) -> bool:
    """Check that string is date-formatted"""
    if not isinstance(s, str):
        return False
    try:
        datetime.datetime.strptime(s, fmt)
        return True
    except Exception as e:
        return False


class ETLRequest():
    """
    Request object for Extract-Transform-Load operations.

    Config for request is specified at time of instantiation with format:
    config = {
        'extract': {
            'filters': {
                <keys: 'video_id', 'username', 'upload_date', 'timestamp_accessed';
                 values: str or List[str]>
            },
            'limit': <int>
        }
    }
    """
    def __init__(self, config: dict):
        # fields
        self.extract: dict = None

        # set fields with validation
        config = self._validate_config(config)
        self.extract = config['extract']

    def _validate_config(self, config: dict) -> dict:
        if 'extract' not in config:
            config['extract'] = {}
        self._validate_config_extract(config['extract'])
        return config

    @staticmethod
    def _validate_config_extract(config_extract: dict):
        # ensure specified options are a subset of valid options
        valid_keys = ['filters', 'limit']
        assert len(set(list(config_extract.keys())) - set(valid_keys)) == 0

        # validate extract filters
        for key, val in config_extract['filters'].items():
            if key == 'video_id':
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            elif key == 'username':
                assert isinstance(val, str)
            elif key == 'upload_date':
                fmt = '%Y-%m-%d'
                assert (is_datetime_formatted_str(val, fmt) or
                        (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            elif key == 'timestamp_accessed':
                fmt = '%Y-%m-%d %H:%M:%S.%f'
                assert (is_datetime_formatted_str(val, fmt) or
                        (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_extract:
            assert isinstance(config_extract['limit'], int)
        else:
            config_extract['limit'] = None




def etl_main(req: ETLRequest):
    """Entry point for ETL preprocessor"""
    data = etl_extract(req)
    etl_transform(data, req)
    return data
    etl_load(data, req)

def etl_extract_tabular(req: ETLRequest) -> Tuple[pd.DataFrame, dict]:
    """Extract tabular data according to request"""
    # construct query components
    database = DB_VIDEOS_DATABASE
    tablename_primary = DB_VIDEOS_TABLENAMES['stats']
    tablename_secondary = DB_VIDEOS_TABLENAMES['meta']
    table_pseudoname_primary = 'stats'
    table_pseudoname_secondary = 'meta'
    join_condition = f'{table_pseudoname_primary}.video_id = {table_pseudoname_secondary}.video_id'

    cols_all = {
        table_pseudoname_primary: ['video_id', 'timestamp_accessed', 'like_count', 'view_count', 'subscriber_count',
                                   'comment_count', 'comment'],
        table_pseudoname_secondary: ['username', 'title', 'upload_date', 'duration', 'keywords', 'description', 'tags']
    }
    cols_for_query = ([f'{table_pseudoname_primary}.{colname}' for colname in cols_all[table_pseudoname_primary]] +
                      [f'{table_pseudoname_secondary}.{colname}' for colname in cols_all[table_pseudoname_secondary]])
    cols_for_df = cols_all[table_pseudoname_primary] + cols_all[table_pseudoname_secondary]

    # where clause
    where_clause = None

    if len(req.extract['filters']) > 0:
        where_clauses = []
        for key, val in req.extract['filters'].items():
            # identify table that this condition applies to
            tablename_ = [tname for tname, colnames_ in cols_all.items() if key in colnames_]
            assert len(tablename_) == 1
            tablename_ = tablename_[0]

            # add where clause
            if isinstance(val, str):  # equality condition
                where_clauses.append(f" {tablename_}.{key} = '{val}'")
            elif isinstance(val, list):  # range condition
                where_clauses.append(f" {tablename_}.{key} BETWEEN '{val[0]}' AND '{val[1]}'")

        # join sub-clauses
        where_clause = ' AND '.join(where_clauses)

    # limit
    limit = req.extract['limit']

    # issue request
    engine = MySQLEngine(DB_CONFIG)
    df = engine.select_records_with_join(
        database,
        tablename_primary,
        tablename_secondary,
        join_condition,
        cols_for_query,
        table_pseudoname_primary=table_pseudoname_primary,
        table_pseudoname_secondary=table_pseudoname_secondary,
        where_clause=where_clause,
        limit=limit,
        cols_for_df=cols_for_df
    )

    # collect info from extraction
    extract_info = dict(
        table_pseudoname_primary=table_pseudoname_primary,
        table_pseudoname_secondary=table_pseudoname_secondary
    )

    return df, extract_info

def etl_extract_nontabular(df: pd.DataFrame,
                           info_tabular_extract: dict) \
        -> Dict[str, Image]:
    """Non-tabular data"""
    video_ids: List[str] = list(df['video_id'].unique())  # get unique video IDs
    records_lst: List[Dict[str, Union[str, bytes]]] = get_mongodb_records(
        DB_NOSQL_DATABASE,
        DB_NOSQL_COLLECTION_NAMES['thumbnails'],
        ids=video_ids
    )
    records: Dict[str, Image] = {e['_id']: convert_bytes_to_image(e['img']) for e in records_lst}
    return records

def etl_extract(req: ETLRequest) -> Dict[str, Union[pd.DataFrame, Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    df, info_tabular_extract = etl_extract_tabular(req)
    records = etl_extract_nontabular(df, info_tabular_extract)
    return dict(
        stats=df,
        images=records
    )


def etl_transform(data: dict,
                  req: ETLRequest):
    """Transform step of ETL pipeline"""
    etl_clean_raw_data(data, req)
    etl_featurize(data, req)

def etl_clean_raw_data(data: dict,
                       req: ETLRequest):
    """
    Clean the raw data
    """
    # fill missing numerical values (zeroes)
    df = data['stats']
    username_video_id_pairs = df[['username', 'video_id']].drop_duplicates()
    for _, (username, video_id) in username_video_id_pairs.iterrows():
        mask = (df['username'] == username) * (df['video_id'] == video_id)
        df[mask] = df[mask].replace(0, np.nan).interpolate(method='linear', axis=0)

    # filter text fields
    a = 5

    # handle missing values

    pass

def etl_featurize(data: dict,
                  req: ETLRequest):
    """Map cleaned raw data to features"""
    pass # add features to data dict

def etl_load(data: dict,
             req: ETLRequest):
    """Load extracted features to feature store"""
    pass