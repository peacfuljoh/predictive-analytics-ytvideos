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

from typing import Any
import datetime
from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.utils.db_mongo_utils import MongoDBEngine


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
    return data
    etl_transform(data, req)
    etl_load(data, req)

def etl_extract(req: ETLRequest) -> dict:
    """Extract step of ETL pipeline"""
    # define tabular data request
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
    cols = ([f'{table_pseudoname_primary}.{colname}' for colname in cols_all[table_pseudoname_primary]] +
            [f'{table_pseudoname_secondary}.{colname}' for colname in cols_all[table_pseudoname_secondary]])

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
            if isinstance(val, str): # equality condition
                where_clauses.append(f" {tablename_}.{key} = '{val}'")
            elif isinstance(val, list): # range condition
                where_clauses.append(f" {tablename_}.{key} BETWEEN '{val[0]}' AND '{val[1]}'")
        where_clause = ' AND '.join(where_clauses)

    # limit
    limit = req.extract['limit']

    # apply request
    engine = MySQLEngine(DB_CONFIG)
    df = engine.select_records_with_join(
        database,
        tablename_primary,
        tablename_secondary,
        join_condition,
        cols,
        table_pseudoname_primary=table_pseudoname_primary,
        table_pseudoname_secondary=table_pseudoname_secondary,
        where_clause=where_clause,
        limit=limit
    )

    return df

    # load non-tabular data


def etl_transform(data: dict,
                  req: ETLRequest):
    """Transform step of ETL pipeline"""
    etl_clean_raw_data(data, req)
    etl_featurize(data, req)

def etl_clean_raw_data(data: dict,
                       req: ETLRequest):
    """
    Clean the raw data:
    - filter text fields
    - handle missing values
    """
    # apply filters

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