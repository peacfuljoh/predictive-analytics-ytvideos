"""Prefeaturization ETL utils"""

import json
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union, Optional, Generator
from pprint import pprint

import numpy as np
import pandas as pd
from PIL import Image

from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import perform_join_mysql_query
from ytpa_utils.val_utils import is_datetime_formatted_str, is_list_of_strings, is_list_of_list_of_time_range_strings
from ytpa_utils.df_utils import get_duplicate_idxs
from ytpa_utils.misc_utils import remove_trailing_chars
from src.crawler.crawler.utils.mongodb_utils_ytvideos import convert_ts_fmt_for_mongo_id
from src.crawler.crawler.constants import (STATS_ALL_COLS, META_ALL_COLS_NO_URL, STATS_NUMERICAL_COLS,
                                           PREFEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_TOKENS_COL, TIMESTAMP_FMT, DATE_FMT,
                                           ETL_CONFIG_VALID_KEYS_PREFEATURES, ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES,
                                           COL_UPLOAD_DATE, COL_VIDEO_ID, COL_USERNAME, COL_LIKE_COUNT,
                                           COL_COMMENT_COUNT, COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT,
                                           COL_TIMESTAMP_ACCESSED, COL_COMMENT, COL_TITLE, COL_KEYWORDS,
                                           COL_DESCRIPTION, COL_TAGS)
from src.etl.etl_request import ETLRequest, req_to_etl_config_record, validate_etl_config
from src.schemas.schema_validation import validate_mongodb_records_schema
from src.schemas.schemas import SCHEMAS_MONGODB


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
        if 'filters' not in config_:
            config_['filters'] = {}
        for key, val in config_['filters'].items():
            if key == COL_VIDEO_ID:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == COL_USERNAME:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == COL_UPLOAD_DATE:
                fmt = DATE_FMT
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            elif key == COL_TIMESTAMP_ACCESSED:
                fmt = TIMESTAMP_FMT
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert config_['limit'] is None or isinstance(config_['limit'], int)
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

    def _validate_config_db(self, config: dict):
        # config keys
        if 0:
            assert set(config) == {'db_info', 'db_mysql_config', 'db_mongo_config'}

            # db info (database/table/collection names, etc.)
            info = config['db_info']
            assert set(info) == {
                'DB_VIDEOS_DATABASE', 'DB_VIDEOS_TABLES',
                'DB_VIDEOS_NOSQL_DATABASE', 'DB_VIDEOS_NOSQL_COLLECTIONS',
                'DB_FEATURES_NOSQL_DATABASE', 'DB_FEATURES_NOSQL_COLLECTIONS',
                'DB_MODELS_NOSQL_DATABASE', 'DB_MODELS_NOSQL_COLLECTIONS'

            }
            assert set(info['DB_VIDEOS_TABLES']) == {'users', 'meta', 'stats'}
            assert set(info['DB_VIDEOS_NOSQL_COLLECTIONS']) == {'thumbnails'}
            assert set(info['DB_FEATURES_NOSQL_COLLECTIONS']) == {'prefeatures', 'features', 'vocabulary',
                                                                   'etl_config_prefeatures', 'etl_config_features',
                                                                   'etl_config_vocabulary'}
            assert set(info['DB_MODELS_NOSQL_COLLECTIONS']) == {'models', 'meta'}

            # db configs (server connection credentials)
            assert set(config['db_mysql_config']) == {'host', 'user', 'password'}
            assert set(config['db_mongo_config']) == {'host', 'port'}






""" ETL Extract """
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


def etl_extract_nontabular(df: pd.DataFrame,
                           info_tabular_extract: dict) \
        -> Dict[str, Image]:
    """Non-tabular raw_data"""
    # TODO: replace call to get_mongodb_records() with call to get_mongodb_records_gen() and iter through gen
    pass
    # video_ids: List[str] = list(df[COL_VIDEO_ID].unique())  # get unique video IDs
    # records_lst: List[Dict[str, Union[str, bytes]]] = get_mongodb_records(
    #     DB_VIDEOS_NOSQL_DATABASE,
    #     DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'],
    #     DB_MONGO_CONFIG,
    #     ids=video_ids
    # )
    # records: Dict[str, Image] = {e['_id']: convert_bytes_to_image(e['img']) for e in records_lst}
    # return records


""" ETL Transform """
def etl_process_title_and_comment(df: pd.DataFrame,
                                  colname: str,
                                  chars: Optional[Union[OrderedDict[str, str], Dict[str, str]]] = None,
                                  charset: Optional[str] = None):
    """
    To print one comment per line:
        for val in df[COL_COMMENT].unique():
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

    colname = COL_KEYWORDS

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

    colname = COL_TAGS

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

    colname = COL_DESCRIPTION

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
        username_video_id_pairs = df[[COL_USERNAME, COL_VIDEO_ID]].drop_duplicates()
        for _, (username, video_id) in username_video_id_pairs.iterrows():
            mask = (df[COL_USERNAME] == username) * (df[COL_VIDEO_ID] == video_id)
            df[mask] = df[mask].replace(0, np.nan).interpolate(method='linear', axis=0, limit_direction='both')

    # filter text fields: COL_COMMENT, COL_TITLE, COL_KEYWORDS, COL_DESCRIPTION, COL_TAGS
    etl_process_title_and_comment(df, COL_COMMENT, charset='LNP')
    etl_process_title_and_comment(df, COL_TITLE, charset='LNP')
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
            # print_df_full(df.loc[df_err.any(axis=1), keys_num + [COL_VIDEO_ID, COL_USERNAME]])
            print(f'Dropping {df_err.any(axis=1).sum()} record(s).')

    return df

def etl_clean_raw_data(data: dict,
                       req: ETLRequestPrefeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Clean the raw raw_data"""
    for df in data['stats']:
        yield etl_clean_raw_data_one_df(df)

def get_words_and_counts(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Count how many unique words exist in the video stats info.

    2023-09-26: {'video_id': 855, 'username': 11, 'title': 3031, 'description': 5007, 'keywords': 2316}
    """
    words = {}
    counts = {}
    for k in [COL_VIDEO_ID, COL_USERNAME]:
        words[k] = list(df[k].unique())
        counts[k] = len(words[k])
    for k in [COL_TITLE, COL_DESCRIPTION]:
        words[k] = list(np.unique([w for e in df[k] if e is not None for w in e.split(' ')]))
        counts[k] = len(words[k])
    for k in [COL_KEYWORDS, COL_TAGS]:
        words[k] = list(np.unique([w for e in df[k] if e is not None for w in json.loads(e)]))
        counts[k] = len(words[k])

    return words, counts

def etl_featurize_make_raw_features(df_gen: Generator[pd.DataFrame, None, None],
                                    req: ETLRequestPrefeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """
    Parse out raw model inputs and outputs from generated DataFrames into raw feature records.
    Text is space-separated word lists.
    """
    # all keys; final key list contains num + meta + tokens + other
    keys_text = [COL_TITLE, COL_KEYWORDS, COL_TAGS]
    keys_num = [COL_SUBSCRIBER_COUNT, COL_COMMENT_COUNT, COL_LIKE_COUNT, COL_VIEW_COUNT]
    keys_meta = [COL_USERNAME, COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED]
    keys_other = req.get_transform()['include_additional_keys'] if 'include_additional_keys' in req.get_transform() else []

    for df in df_gen:
        df[PREFEATURES_TOKENS_COL] = df[keys_text].agg(' '.join, axis=1)
        yield df[list(set([PREFEATURES_TOKENS_COL] + keys_num + keys_meta + keys_other))]

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

        colname = COL_TITLE
        idxs = get_duplicate_idxs(df, colname)
        for i, idxs_ in idxs.items():
            row: pd.Series = df.loc[i]
            print(', '.join([row[key] for key in [COL_TITLE, COL_KEYWORDS, COL_TAGS]]))

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
    """
    cols_exclude = [COL_USERNAME, COL_VIDEO_ID]

    records_all: List[dict] = [] # update cmds
    for video_id, df_ in df.groupby(COL_VIDEO_ID):
        # get info for DB update
        cols_include = [col for col in df_.columns if col not in cols_exclude]
        username = df_.iloc[0][COL_USERNAME] # should be only one username
        d_records = df_.loc[:, cols_include].set_index(COL_TIMESTAMP_ACCESSED).to_dict('index') # re-index by timestamp

        # set update info
        for ts, rec in d_records.items():
            # convert ts format
            ts_str, ts_str_id = convert_ts_fmt_for_mongo_id(ts)

            # define record for insertion
            record_to_insert = {
                '_id': f'{video_id}_{ts_str_id}_{req.name}', # required to avoid duplication
                COL_USERNAME: username,
                COL_VIDEO_ID: video_id,
                COL_TIMESTAMP_ACCESSED: ts_str,
                PREFEATURES_ETL_CONFIG_COL: req.name,
                **rec
            }
            records_all.append(record_to_insert)

    assert validate_mongodb_records_schema(records_all, SCHEMAS_MONGODB['prefeatures'])

    return records_all

def etl_load_prefeatures(data: Dict[str, Generator[pd.DataFrame, None, None]],
                         req: ETLRequestPrefeatures):
    """Load prefeatures to NoSQL database."""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_config = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_prefeatures']
    collection_prefeatures = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['prefeatures']

    # insert etl_config
    engine = MongoDBEngine(mongo_config,
                           database=database,
                           collection=collection_config,
                           verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # insert records
    engine = MongoDBEngine(mongo_config,
                           database=database,
                           collection=collection_prefeatures,
                           verbose=True)
    for df in data['stats']:
        print(f"etl_load_prefeatures() -> Inserting {len(df)} records in collection {collection_prefeatures} "
              f"of database {database}")
        records = etl_load_prefeatures_prepare_for_insert(df, req)
        engine.insert_many(records)
