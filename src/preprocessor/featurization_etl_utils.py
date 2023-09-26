"""Featurization ETL util methods"""

import json
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union, Optional
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

from src.crawler.crawler.config import DB_INFO, DB_CONFIG
from src.crawler.crawler.utils.db_mongo_utils import get_mongodb_records
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.utils.misc_utils import convert_bytes_to_image, is_datetime_formatted_str
from src.crawler.crawler.constants import STATS_ALL_COLS, META_ALL_COLS_NO_URL, STATS_NUMERICAL_COLS


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']
DB_NOSQL_DATABASE = DB_INFO['DB_NOSQL_DATABASE']
DB_NOSQL_COLLECTION_NAMES = DB_INFO['DB_NOSQL_COLLECTION_NAMES']

CHARSETS = {
    'LNP': 'abcdefghijklmnopqrstuvwxyz1234567890.?!$\ ',
    'LN': 'abcdefghijklmnopqrstuvwxyz1234567890',
}

MIN_DESCRIPTION_LEN_0 = 20
MIN_DESCRIPTION_LEN_1 = 20


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
        table_pseudoname_primary: STATS_ALL_COLS,
        table_pseudoname_secondary: [col for col in META_ALL_COLS_NO_URL if col != 'video_id']
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

def remove_trailing_chars(s: str) -> str:
    """Remove trailing chars from a string (e.g. newlines, empty spaces)."""
    if len(s) == 0:
        return s
    trail_chars = ['\n', ' ']
    i = len(s) - 1
    while (i > 0) and (s[i] in trail_chars):
        i -= 1
    return s[:i]

def etl_process_title_and_comment(df: pd.DataFrame,
                                  colname: str,
                                  chars: Optional[Union[OrderedDict[str, str], Dict[str, str]]] = None,
                                  charset: Optional[str] = None):
    """
    To print one comment per line:
        for val in df['comment'].unique():
            print(val)
    """
    assert isinstance(df, pd.DataFrame)

    idxs = get_duplicate_idxs(df, colname)

    if chars is None:
        chars = OrderedDict()
        chars[r'\"'] = ''
        chars['...'] = ' '
        chars['..'] = ' '
        chars['.'] = ''
        chars['-'] = ' '
        chars["’s"] = ''
        chars['?'] = ' ?'
        chars['!'] = ' !'
        chars['$'] = ' $'
        chars['\n'] = ' '

    for i, idxs_ in idxs.items():
        s: str = df.loc[i, colname]
        for k, v in chars.items():
            s = s.replace(k, v)
        if charset is not None:
            s = ''.join([c for c in s if c.lower() in CHARSETS[charset]])
        s = remove_trailing_chars(s)
        s = s.lower()
        df.loc[idxs_, colname] = s

def etl_process_keywords(df: pd.DataFrame):
    """Process raw keywords"""
    assert isinstance(df, pd.DataFrame)

    colname = 'keywords'

    str_repl = OrderedDict()
    str_repl['#'] = ''
    str_repl['...'] = ' '
    str_repl['..'] = ' '
    str_repl['.'] = ''
    str_repl['-'] = ' '
    str_repl["’s"] = ''
    str_repl["'s"] = ''
    str_repl['?'] = ''
    str_repl['!'] = ''
    str_repl['$'] = ''
    str_repl['('] = ''
    str_repl[')'] = ''

    idxs = get_duplicate_idxs(df, colname)

    for i, idxs_ in idxs.items():
        s: str = df.loc[i, colname]
        s: List[str] = json.loads(s)
        s_new: List[str] = []
        for keyphrase in s: # str in List[str]
            for keyword in keyphrase.split(' '): # str in List[str]
                if '__' in keyword or ':' in keyword or len(keyword) < 2:
                    continue
                keyword = keyword.lower()
                for key, val in str_repl.items():
                    keyword = keyword.replace(key, val)
                if ' ' in keyword: # spaces were inserted
                    s_new += keyword.split(' ')
                else:
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

def etl_process_description(df: pd.DataFrame):
    """Process raw description"""
    assert isinstance(df, pd.DataFrame)

    colname = 'description'

    str_repl = OrderedDict()
    str_repl['...'] = ' '
    str_repl['..'] = ' '
    str_repl['.'] = ''
    str_repl['-'] = ' '
    str_repl['?'] = ' ?'
    str_repl['!'] = ' !'
    str_repl['$'] = ' $'

    idxs = get_duplicate_idxs(df, colname)

    for i, idxs_ in idxs.items():
        # get entry and loads to string
        s: str = df.loc[i, colname]

        # accumulate valid entries
        s_new: List[str] = []
        for line in s.split("\n"):
            # skip if invalid
            if (('http' in line) or ('@' in line) or (len(line) <= MIN_DESCRIPTION_LEN_0)):
                continue

            # remove special characters
            line = ''.join([c for c in line if c.lower() in CHARSETS['LNP']])
            if len(line) <= MIN_DESCRIPTION_LEN_1:
                continue
            # print('----'); print(line); print(len(line))

            # cast to lower, replace substrings
            line = line.lower()
            for key, val in str_repl.items():
                line = line.replace(key, val)

            # remove trailing spaces
            line = remove_trailing_chars(line)

            # add to list
            s_new.append(line)

        # filter out duplicates, dump back to string
        s = None if len(s_new) == 0 else ' '.join(s_new)
        # print(s_new); print(len(s_new)); print(s)
        df.loc[idxs_, colname] = s

def etl_clean_raw_data(data: dict,
                       req: ETLRequest):
    """Clean the raw data"""
    # fill missing numerical values (zeroes)
    df = data['stats']
    username_video_id_pairs = df[['username', 'video_id']].drop_duplicates()
    for _, (username, video_id) in username_video_id_pairs.iterrows():
        mask = (df['username'] == username) * (df['video_id'] == video_id)
        df[mask] = df[mask].replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')

    # filter text fields: 'comment', 'title', 'keywords', 'description', 'tags'
    etl_process_title_and_comment(df, 'comment', charset='LNP')
    etl_process_title_and_comment(df, 'title', charset='LNP')
    etl_process_keywords(df)
    etl_process_tags(df)
    etl_process_description(df)

    # handle missing values
    literals_err = [np.NaN, 0, None]
    df_err = df[STATS_NUMERICAL_COLS].isin(literals_err)
    err_mask_sum: pd.Series = df_err.sum(axis=0)
    if err_mask_sum.sum() > 0:
        err_msg = f'\netl_clean_raw_data() -> Some numerical data entries are invalid. Total count is {len(df)}; ' + \
                   ', '.join([f'{key}: {val}' for key, val in err_mask_sum.items()])
        print(err_msg)
        # print_df_full(df.loc[df_err.any(axis=1), keys_num + ['video_id', 'username']])
        print(f'Dropping {df_err.any(axis=1).sum()} record(s).')


def get_words_and_counts(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Count how many unique words exist in the video stats info.

    2023-09-26: {'video_id': 855, 'username': 11, 'title': 3031, 'description': 5007, 'keywords': 2316}
    """
    words = {}
    counts = {}
    for k in ['video_id', 'username']:
        words[k] = list(df[k].unique())
        counts[k] = len(words[k])
    for k in ['title', 'description']:
        words[k] = list(np.unique([w for e in df[k] if e is not None for w in e.split(' ')]))
        counts[k] = len(words[k])
    for k in ['keywords', 'tags']:
        words[k] = list(np.unique([w for e in df[k] if e is not None for w in json.loads(e)]))
        counts[k] = len(words[k])

    return words, counts

def etl_featurize(data: dict,
                  req: ETLRequest):
    """Map cleaned raw data to features"""
    pprint(get_words_and_counts(data['stats'])[1])
    a = 5

    # featurize text via embedding into Vector Space Model (VSM)
    # - Gensim (Python library for NLP)


    # featurize thumbnail images


    a = 5 # add features to data dict
