"""Prefeaturization ETL utils"""

import json
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union, Optional, Generator
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine
from src.crawler.crawler.utils.mysql_engine import MySQLEngine
from src.crawler.crawler.utils.misc_utils import (is_datetime_formatted_str,
                                                  df_generator_wrapper, is_list_of_strings,
                                                  is_list_of_list_of_time_range_strings)
from src.crawler.crawler.utils.mongodb_utils_ytvideos import convert_ts_fmt_for_mongo_id
from src.crawler.crawler.constants import (STATS_ALL_COLS, META_ALL_COLS_NO_URL, STATS_NUMERICAL_COLS,
                                           PREFEATURES_USERNAME_COL, PREFEATURES_TIMESTAMP_COL,
                                           PREFEATURES_VIDEO_ID_COL, PREFEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_TOKENS_COL, TIMESTAMP_FMT)
from src.etl.etl_request import ETLRequest, req_to_etl_config_record

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE'] # tabular raw
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']
DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE'] # NoSQL thumbnails
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']

CHARSETS = {
    'LNP': 'abcdefghijklmnopqrstuvwxyz1234567890.?!$\ ',
    'LN': 'abcdefghijklmnopqrstuvwxyz1234567890',
}

MIN_DESCRIPTION_LEN_0 = 20
MIN_DESCRIPTION_LEN_1 = 20

""" ETL request class for prefeatures processing """
class ETLRequestPrefeatures(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        for key, val in config_['filters'].items():
            if key == 'video_id':
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == 'username':
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == 'upload_date':
                fmt = '%Y-%m-%d'
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            elif key == 'timestamp_accessed':
                fmt = TIMESTAMP_FMT
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

    def _validate_config_transform(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'transform')

        # validate transform filters
        # ...

        # validate other transform options
        if 'include_additional_keys' in config_:
            assert isinstance(config_['include_additional_keys'], list)
            assert is_list_of_strings(list(config_['include_additional_keys']))




""" ETL Extract """
def etl_extract_tabular(req: ETLRequestPrefeatures) -> Tuple[Generator[pd.DataFrame, None, None], dict]:
    """Extract tabular raw_data according to request"""
    # construct query components
    database = DB_VIDEOS_DATABASE
    tablename_primary = DB_VIDEOS_TABLES['stats']
    tablename_secondary = DB_VIDEOS_TABLES['meta']
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

    if len(req.get_extract()['filters']) > 0:
        where_clauses = []
        for key, val in req.get_extract()['filters'].items():
            # identify table that this condition applies to
            tablename_ = [tname for tname, colnames_ in cols_all.items() if key in colnames_]
            assert len(tablename_) == 1
            tablename_ = tablename_[0]

            # add where clause
            if isinstance(val, str):  # equality
                where_clauses.append(f" {tablename_}.{key} = '{val}'")
            elif isinstance(val, list): # range or subset
                if isinstance(val[0], list): # range
                    where_clauses.append(f" {tablename_}.{key} BETWEEN '{val[0][0]}' AND '{val[0][1]}'")
                else: # subset
                    val_strs = [f"'{s}'" for s in val] # add apostrophes for SQL query with strings
                    where_clauses.append(f" {tablename_}.{key} IN ({','.join(val_strs)})")

        # join sub-clauses
        where_clause = ' AND '.join(where_clauses)

    # limit
    limit = req.get_extract()['limit']

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
        cols_for_df=cols_for_df,
        as_generator=True
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
    """Non-tabular raw_data"""
    # TODO: replace call to get_mongodb_records() with call to get_mongodb_records_gen() and iter through gen
    pass
    # video_ids: List[str] = list(df['video_id'].unique())  # get unique video IDs
    # records_lst: List[Dict[str, Union[str, bytes]]] = get_mongodb_records(
    #     DB_VIDEOS_NOSQL_DATABASE,
    #     DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'],
    #     DB_MONGO_CONFIG,
    #     ids=video_ids
    # )
    # records: Dict[str, Image] = {e['_id']: convert_bytes_to_image(e['img']) for e in records_lst}
    # return records


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
    while (i >= 0) and (s[i] in trail_chars):
        i -= 1
    return s[:i + 1]

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
        df.loc[idxs_, colname] = ' '.join(s)


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
        df.loc[idxs_, colname] = ' '.join(s)

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

def etl_clean_raw_data_one_df(df: pd.DataFrame):
    """Clean one raw dataframe"""
    # interpolate and extrapolate missing values (zeroes and NaNs)
    if 0:
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
    if 0:
        literals_err = [np.NaN, None]
        df_err = df[STATS_NUMERICAL_COLS].isin(literals_err)
        err_mask_sum: pd.Series = df_err.sum(axis=0)
        if err_mask_sum.sum() > 0:
            err_msg = f'\netl_clean_raw_data_one_df() -> Some numerical raw_data entries are invalid. Total count is {len(df)}; ' + \
                      ', '.join([f'{key}: {val}' for key, val in err_mask_sum.items()])
            print(err_msg)
            # print_df_full(df.loc[df_err.any(axis=1), keys_num + ['video_id', 'username']])
            print(f'Dropping {df_err.any(axis=1).sum()} record(s).')

    return df


@df_generator_wrapper
def etl_clean_raw_data(data: dict,
                       req: ETLRequestPrefeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Clean the raw raw_data"""
    df = etl_clean_raw_data_one_df(next(data['stats']))
    return df

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

@df_generator_wrapper
def etl_featurize_make_raw_features(df_gen: Generator[pd.DataFrame, None, None],
                                    req: ETLRequestPrefeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """
    Parse out raw model inputs and outputs from generated DataFrames into raw feature records.
    Text is space-separated word lists.
    """
    # all keys; final key list contains num + meta + tokens + other
    keys_text = ['title', 'keywords', 'tags']
    keys_num = ['subscriber_count', 'comment_count', 'like_count', 'view_count']
    keys_meta = ['username', 'video_id', 'timestamp_accessed']
    keys_other = req.get_transform()['include_additional_keys'] if 'include_additional_keys' in req.get_transform() else []

    df = next(df_gen)
    if df.empty:
        raise StopIteration
    df[PREFEATURES_TOKENS_COL] = df[keys_text].agg(' '.join, axis=1)
    return df[list(set([PREFEATURES_TOKENS_COL] + keys_num + keys_meta + keys_other))]

def etl_featurize(data: Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]],
                  req: ETLRequestPrefeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Map cleaned raw raw_data to features"""
    # inspect words lists and counts
    if 0:
        dfs = []
        while not (df_ := next(data['stats'])).empty:
            dfs.append(df_)
        df = pd.concat(dfs, axis=0, ignore_index=True)
        wc = get_words_and_counts(df)
        words_all = [word for key, val in wc[0].items() for word in val]
        print(len(words_all))
        words_all_unique = np.unique(words_all)
        print(len(words_all_unique))

        pprint(wc[1])

        colname = 'title'
        idxs = get_duplicate_idxs(df, colname)
        for i, idxs_ in idxs.items():
            row: pd.Series = df.loc[i]
            print(', '.join([row[key] for key in ['title', 'keywords', 'tags']]))

    # featurize text via embedding into Vector Space Model (VSM)
    raw_features = etl_featurize_make_raw_features(data['stats'], req)

    # featurize thumbnail images
    # TODO: implement this

    return raw_features



""" ETL Load """
def etl_load_prefeatures_prepare_for_insert(df: pd.DataFrame,
                                            req: ETLRequestPrefeatures) \
        -> List[dict]:
    """
    Convert DataFrame with prefeatures info into dict for MongoDB insertion.

    Format for each document:
    {
        '_id': ...,
        'username': ...,
        'timestamp_accessed': ...,
        'video_id': ...,
        'etl_config': ...,
        'raw_data': ...
    }
    """
    cols_exclude = ['username', 'video_id']
    cols_include = None

    records_all: List[dict] = [] # update cmds
    for video_id, df_ in df.groupby('video_id'):
        # get info for DB update
        if cols_include is None:
            cols_include = [col for col in df_.columns if col not in cols_exclude]
        username = df_.iloc[0]['username'] # should be only one username
        d_records = df_.loc[:, cols_include].set_index('timestamp_accessed').to_dict('index') # re-index by timestamp

        # set update info
        for ts, rec in d_records.items():
            # convert ts format
            ts_str, ts_str_id = convert_ts_fmt_for_mongo_id(ts)

            # define record for insertion
            record_to_insert = {
                '_id': f'{video_id}_{ts_str_id}_{req.name}', # required to avoid duplication
                PREFEATURES_USERNAME_COL: username,
                PREFEATURES_TIMESTAMP_COL: ts_str,
                PREFEATURES_VIDEO_ID_COL: video_id,
                PREFEATURES_ETL_CONFIG_COL: req.name,
                **rec
            }
            records_all.append(record_to_insert)

    return records_all

def etl_load_prefeatures(data: Dict[str, Generator[pd.DataFrame, None, None]],
                         req: ETLRequestPrefeatures):
    """Load prefeatures to NoSQL database."""
    # insert etl_config
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_prefeatures'],
                           verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # insert records
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
                           verbose=True)
    while not (df := next(data['stats'])).empty:
        records = etl_load_prefeatures_prepare_for_insert(df, req)
        engine.insert_many(records)
