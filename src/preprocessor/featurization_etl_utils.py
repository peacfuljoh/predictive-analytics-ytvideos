import json
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union, Optional

import pandas as pd
from PIL import Image

from src.crawler.crawler.config import DB_INFO, DB_CONFIG
from src.crawler.crawler.utils.db_mongo_utils import get_mongodb_records
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.utils.misc_utils import convert_bytes_to_image, is_datetime_formatted_str

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']
DB_NOSQL_DATABASE = DB_INFO['DB_NOSQL_DATABASE']
DB_NOSQL_COLLECTION_NAMES = DB_INFO['DB_NOSQL_COLLECTION_NAMES']

CHARSETS = {
    'LNP': 'abcdefghijklmnopqrstuvwxyz1234567890.?!$\ '
}


""" ETL Request class """
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


""" ETL Extract """
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


""" ETL Transform """
def get_duplicate_idxs(df: pd.DataFrame,
                       colname: str) \
        -> pd.DataFrame:
    """
    Get duplicate indices for entries in a specified column of a DataFrame.

    Steps:
        - adds col with index values
        - group rows by specified column
        - aggregate rows into groups, add two cols with duplicate first appearance + row indices where duplicates appear
        - convert to DataFrame with index (first index) and one column (duplicate indices)
    """
    idxs = (df[[colname]].reset_index()
            .groupby([colname])['index']
            .agg(['first', tuple])
            .set_index('first')['tuple'])
    return idxs

def replace_chars_in_str(df: pd.DataFrame,
                         colname: str,
                         chars: Optional[Union[OrderedDict[str, str], Dict[str, str]]] = None,
                         charset: Optional[str] = None):
    """
    Replace specified substrings in strings of specified column.

    To print one comment per line:
        for val in df['comment'].drop_duplicates().values:
            print(val)
    """
    assert isinstance(df, pd.DataFrame)

    idxs = get_duplicate_idxs(df, colname)

    if chars is None:
        chars = OrderedDict()
        chars[r'\"'] = ''
        chars['...'] = ' '
        chars['..'] = ' '
        chars['-'] = ' '
        chars["’s"] = ''

    for i, idxs_ in idxs.items():
        s: str = df.loc[i, colname]
        for k, v in chars.items():
            s = s.replace(k, v)
        if charset is not None:
            s = ''.join([c for c in s if c.lower() in CHARSETS[charset]])
        df.loc[idxs_, colname] = s

def etl_process_keywords(df: pd.DataFrame):
    """Process raw keywords"""
    assert isinstance(df, pd.DataFrame)

    colname = 'keywords'

    str_repl = OrderedDict()
    str_repl['#'] = ''

    idxs = get_duplicate_idxs(df, colname)

    for i, idxs_ in idxs.items():
        s: str = df.loc[i, colname]
        s: List[str] = json.loads(s)
        s_new: List[str] = []
        for keyphrase in s: # str in List[str]
            for keyword in keyphrase.split(' '): # str in List[str]
                if '__' in keyword or len(keyword) < 2:
                    continue
                keyword = keyword.lower()
                for key, val in str_repl.items():
                    keyword = keyword.replace(key, val)
                s_new.append(keyword)
        s: List[str] = list(set(s_new))
        df.loc[idxs_, colname] = json.dumps(s)


def etl_process_tags(df: pd.DataFrame):
    """Process raw tags"""
    assert isinstance(df, pd.DataFrame)

    colname = 'tags'

    str_repl = OrderedDict()
    str_repl[' '] = ''
    str_repl[':'] = ''
    str_repl['"'] = ''
    str_repl['\\'] = ''

    idxs = get_duplicate_idxs(df, colname)

    for i, idxs_ in idxs.items():
        # get entry and loads to string
        s: str = df.loc[i, colname]
        s: List[str] = json.loads(s)

        # accumulate valid entries
        s_new: List[str] = []
        for tag_str in s:  # str in List[str]
            # weed out multi-space entries
            if sum([len(p) > 0 for p in tag_str.split(' ')]) > 2:
                continue

            # check each entry
            for tag in re.split(r"\n|\\n|\\", tag_str):  # str in List[str]
                # skip if invalid
                if (('__' in tag) or ('~' in tag) or ('https' in tag) or ('\\' in tag)
                        or ('\\u' in tag) or (len(tag) < 2)):
                    continue

                # cast to lower, replace substrings
                tag = tag.lower()
                for key, val in str_repl.items():
                    tag = tag.replace(key, val)

                # add to list
                s_new.append(tag)

        # filter out duplicates, dump back to string
        s: List[str] = list(set(s_new))
        df.loc[idxs_, colname] = json.dumps(s)

