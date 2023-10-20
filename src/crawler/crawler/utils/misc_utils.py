"""Miscellaneous utils"""

import json
import re
from typing import Union, List, Dict, Optional, Any, Sequence, Callable
import datetime
import time
from datetime import timedelta
import copy
from PIL import Image
import requests
import io

import pandas as pd
import numpy as np


def save_json(path: str,
              obj: Union[List[dict], dict],
              mode: str = 'w'):
    with open(path, mode) as fp:
        json.dump(obj, fp, indent=4)

def load_json(fpath: str) -> Union[dict, list]:
    with open(fpath, 'r') as fp:
        return json.load(fp)

def flatten_dict_of_lists(d: Dict[str, list]) -> list:
    """Convert a dict of lists of elems into a flat list of those elems"""
    return [elem for nested_list in d.values() for elem in nested_list]

def make_videos_page_urls_from_usernames(names: List[str]) -> List[str]:
    """Make YouTube video page URLs for a list of usernames"""
    return [f"https://www.youtube.com/@{name}/videos" for name in names]

def make_video_urls(video_ids: List[str]) -> List[str]:
    return [f"https://www.youtube.com/watch?v={video_id}" for video_id in video_ids]

def convert_num_str_to_int(s: str) -> int:
    """
    Convert number strings to integers. Assumes resulting number is an integer. Handles strings of the form:
    - 535
    - 54,394
    - 52.3K
    - 3.8M
    - 40M
    """
    if s == '':
        return 0
    s = s.replace(',', '')
    if 'K' in s:
        s = int(float(s[:-1]) * 1e3)
    elif 'M' in s:
        s = int(float(s[:-1]) * 1e6)
    num = int(s)
    return num


def apply_regex(s: str,
                regex: str,
                dtype: Optional[str] = None) \
        -> Union[List[str], str, int]:
    """
    Apply regex to string to extract a string. Can handle further parsing if string is a number.
    Assumes regex is specified as string with embedded (.*?) to find substring.
    Handles commas separating sequences of digits (e.g. 12,473).
    """
    substring_flag = '(.*?)'
    assert substring_flag in regex
    res = re.findall(regex, s)
    # print(res)
    num_substrings = sum([regex[i:i + len(substring_flag)] == substring_flag for i in range(len(regex) - len(substring_flag))])
    if num_substrings > 1:
        return res
    if len(res) == 0:
        substring = ''
    else:
        substring = res[0]
    # print(substring)
    if dtype == 'int':
        return convert_num_str_to_int(substring)
    return substring

def get_ts_now_str(mode: str,
                   offset: Optional[datetime.timedelta] = None) \
        -> str:
    """Get current timestamp"""
    def get_ts_now_formatted(fmt: str) -> str:
        ts_now = datetime.datetime.fromtimestamp(time.time())
        if offset is not None:
            return (ts_now + offset).strftime(fmt)
        else:
            return ts_now.strftime(fmt)

    assert mode in ['date', 's', 'ms', 'us']

    if mode == 'date':
        fmt = '%Y-%m-%d'
    elif mode == 's':
        fmt = '%Y-%m-%d %H:%M:%S'
    elif mode in ['ms', 'us']:
        fmt = '%Y-%m-%d %H:%M:%S.%f'
    else:
        raise NotImplementedError

    ts: str = get_ts_now_formatted(fmt)

    if mode == 'ms':
        ts = ts[:-3] # trim last 3 fractional digits

    return ts

def print_df_full(df: pd.DataFrame,
                  row_lims: Optional[List[int]] = None):
    """Print all rows and columns of a dataframe"""
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.expand_frame_repr', False):
        if row_lims is None:
            print(df)
        else:
            print(df[row_lims[0]:row_lims[1]])

def get_dt_now() -> datetime.datetime:
    """Get current datetime"""
    return datetime.datetime.fromtimestamp(time.time())

class TimeLock():
    """
    Utility for managing timed events. TimeLock forces a waiting period until the next point in time an integer multiple
    of intervals ahead of an initial starting time. This allows for triggering events at strict intervals where missed
    intervals are skipped.
    """
    def __init__(self,
                 dt_start: datetime.datetime,
                 interval: int, # interval between lock releases
                 progress_dur: Optional[int] = None, # how often to print update on waiting period (seconds)
                 verbose: bool = False):
        assert (dt_start - get_dt_now()).total_seconds() > 0

        self._dt_target = dt_start
        self._interval = interval
        self._progress_dur = progress_dur
        self._verbose = verbose

    def _wait_until_target(self):
        """Wait until current time catches up to target time."""
        while (t_wait := (self._dt_target - get_dt_now()).total_seconds()) > 0:
            if self._verbose:
                print(f'TimeLock: Waiting {t_wait} seconds until {self._dt_target}.')
            if self._progress_dur is not None:
                t_wait = min(t_wait, self._progress_dur)
            time.sleep(t_wait)

    def _advance_target(self):
        """Advance target time beyond the current time by the minimum integer multiple of the interval."""
        dt_now = get_dt_now()
        dt_target_orig = copy.copy(self._dt_target)
        t_elapsed = (dt_now - self._dt_target).total_seconds()
        if t_elapsed <= 0:
            return
        num_intervals_advance = int(t_elapsed / self._interval) + 1
        self._dt_target += timedelta(seconds=num_intervals_advance * self._interval)
        if self._verbose:
            print(f'TimeLock: Advancing time lock target from {dt_target_orig} to {self._dt_target}.')

    def acquire(self):
        """Acquire lock (i.e. advance target time in preparation for next release)"""
        self._advance_target()
        self._wait_until_target()

def fetch_data_at_url(url: str,
                      delay: float = 0) \
        -> bytes:
    """Fetch raw_data at specified URL"""
    if delay > 0:
        time.sleep(delay)
    return requests.get(url).content

def convert_bytes_to_image(image_data: bytes) -> Image:
    """Convert byte string to Image"""
    return Image.open(io.BytesIO(image_data))




if __name__ == '__main__':
    if 1:
        print(get_ts_now_str(mode='date'))
        print(get_ts_now_str(mode='s'))
        print(get_ts_now_str(mode='ms'))
        print(get_ts_now_str(mode='us'))


def is_datetime_formatted_str(s: Any, fmt: str) -> bool:
    """Check that string is date-formatted"""
    if not isinstance(s, str): # possibly redundant, but safer
        return False
    try:
        datetime.datetime.strptime(s, fmt)
        return True
    except Exception as e:
        return False


def df_generator_wrapper(func):
    """
    Wrapper for handling StopIteration and other generator exceptions when yielding a DataFrame.
    Use this to decorate functions that return a DataFrame.
    """
    def wrap(*args, **kwargs):
        while 1:
            try:
                yield func(*args, **kwargs)
            except StopIteration:
                yield pd.DataFrame()
    return wrap


def is_list_of_strings(obj: Sequence) -> bool:
    """Check that object is a list of strings."""
    return isinstance(obj, list) and all([isinstance(e, str) for e in obj])

def is_list_of_floats(obj: Sequence) -> bool:
    """Check if object is a list of floats."""
    return isinstance(obj, list) and all([isinstance(val, float) for val in obj])

def is_list_of_list_of_strings(obj: Sequence) -> bool:
    """Check that iterable is a list of lists of strings."""
    return isinstance(obj, list) and all([is_list_of_strings(lst) for lst in obj])

def is_list_of_formatted_strings(obj: Sequence,
                                 fmt_check_func: Callable,
                                 list_len: Optional[int] = None) -> bool:
    """Check that iterable is list of formatted strings (formatting checked by provided function)."""
    if isinstance(obj, list):
        len_cond = True if list_len is None else len(obj) == list_len
        return len_cond and all([fmt_check_func(e) for e in obj])
    return False

def is_list_of_list_of_time_range_strings(obj: Sequence,
                                          func: Callable,
                                          num_ranges: Optional[int] = None) \
        -> bool:
    """Check that object is of the form [[<date_or_timestamp_string>, <date_or_timestamp_string>], ...]."""
    if isinstance(obj, list):
        len_cond = True if num_ranges is None else len(obj) == num_ranges
        return len_cond and all([is_list_of_formatted_strings(lst, func, list_len=2) for lst in obj])
    return False

def is_subset(obj1: Sequence,
              obj2: Sequence) \
        -> bool:
    """
    Check that the elements in one iterable comprise a subset of the elements in the other.

    For example:
        assert_subset([0, 1, 2], [0, 4, 2, 1]) returns True
        assert_subset([0, 1, 2], [3, 2, 1, 5]) returns False
    """
    return len(set(obj1) - set(obj2)) == 0

def is_list_of_sequences(obj: Sequence,
                         seq_types: tuple,
                         len_: Optional[int] = None) \
        -> bool:
    """
    Check that object is a list of an iterable type (e.g. tuple, list), optionally checking the length of each entry.
    """
    if isinstance(obj, list):
        for e in obj:
            valid = isinstance(e, seq_types) and (True if len_ is None else len(e) == len_)
            if not valid:
                return False
        return True
    return False

def is_dict_of_instances(obj: Sequence,
                         type) \
        -> bool:
    """Check that object is a dict of objects of a specified instance."""
    return isinstance(obj, dict) and all([isinstance(val, type) for key, val in obj.items()])


def join_on_dfs(df0: pd.DataFrame,
                df1: pd.DataFrame,
                index_keys: List[str],
                df0_keys_select: Optional[List[str]] = None,
                df1_keys_select: Optional[List[str]] = None) \
        -> pd.DataFrame:
    """
    Combine info from two DataFrames analogously to the SQL op:
        SELECT df0.col00, df0.col01, df1.col10 FROM df0 JOIN ON df0.index_key0 = df1.index_key0

    The index keys in df0 are treated like foreign keys into df1.
    """
    # TODO: handle case where index keys in df0 don't exist in df1
    # perform select on df0
    if df0_keys_select is not None:
        df0_select = df0[df0_keys_select]
    else:
        df0_select = df0

    # turn index keys in df1 into multi-index
    df1_mindex = df1.set_index(index_keys, drop=True)

    # get index keys for all rows of df0
    ids_ = df0[index_keys].to_numpy().tolist()
    if len(index_keys) == 1:
        ids_ = [id_[0] for id_ in ids_]
    else:
        ids_ = [tuple(id_) for id_ in ids_]

    # perform select on df1
    assert is_subset(ids_, df1_mindex.index) # all ID keys in df0 must exist in df1's multi-index for the JOIN
    if df1_keys_select is not None:
        df1_select = df1_mindex.loc[ids_, df1_keys_select]
    else:
        df1_select = df1_mindex.loc[ids_]
    df1_select = df1_select.set_index(df0.index)

    # concatenate
    df = pd.concat((df0_select, df1_select), axis=1)

    return df

def convert_mixed_df_to_array(df: pd.DataFrame,
                              cols: Optional[List[str]] = None) \
        -> np.ndarray:
    """
    Convert DataFrame with mixed-type columns into a numpy array.
    Only converts numerical columns. Emits warning for non-numerical/list-type columns.
    """
    if cols is None:
        cols = df.columns

    data: List[np.ndarray] = []
    for col in cols:
        data_ = df[col]
        samp0 = data_.iloc[0]
        if isinstance(samp0, (int, float, np.int64, np.float64)):
            data.append(data_.to_numpy()[:, np.newaxis])
        elif isinstance(samp0, list):
            data.append(np.array(list(data_)))
        else:
            print(f'convert_mixed_df_to_array() -> Skipping column {col} with invalid raw_data type {type(samp0)}.')

    return np.hstack(data)

def just_dict_keys(obj: dict) -> Union[dict, None]:
    """Return dictionary with only keys. Leaf values will be replaced with None."""
    if not isinstance(obj, dict):
        return None

    obj_keys = {}
    for key, val in obj.items():
        obj_keys[key] = just_dict_keys(val)

    return obj_keys


def make_sql_query(tablename: str,
                   cols: Optional[List[str]] = None,
                   where: Optional[Dict[str, Union[str, List[str], List[List[str]]]]] = None,
                   limit: Optional[int] = None) \
        -> str:
    """
    Make a basic SQL query.

    WHERE clause entries are specified for conjunction only (e.g. <cond0> AND <cond1> AND ...). Each entry can be
    a single string (for equality condition), a list of strings (for set condition) or a list of 1 length-2 list with
    range endpoints (for datetime ranges).
    """
    assert isinstance(tablename, str)
    assert cols is None or is_list_of_strings(cols)
    assert (where is None or
            all([
                isinstance(val, str) or
                is_list_of_strings(val) or
                (isinstance(val, list) and (len(val) == 1) and is_list_of_strings(val[0]) and (len(val[0]) == 2))
                for val in where.values()
            ])
            )
    assert limit is None or isinstance(limit, int)

    # base query
    if cols is not None:
        keys_str = ','.join(cols)
    else:
        keys_str = '*'
    query = f'SELECT {keys_str} FROM {tablename}'

    # where
    if where is not None:
        where_clauses = []
        for key, val in where.items():
            where_clauses.append(make_sql_query_where_one(tablename, key, val))
        query += ' WHERE ' + ' AND '.join(where_clauses)

    # limit
    if limit is not None:
        query += f' LIMIT {limit}'

    return query

def make_sql_query_where_one(tablename: str,
                             key: str,
                             val: str) \
        -> str:
    """Create a single where clause entry"""
    if isinstance(val, str):  # equality
        return f" {tablename}.{key} = '{val}'"
    elif isinstance(val, list):  # range or subset
        if isinstance(val[0], list):  # range
            return f" {tablename}.{key} BETWEEN '{val[0][0]}' AND '{val[0][1]}'"
        else:  # subset
            val_strs = [f"'{s}'" for s in val]  # add apostrophes for SQL query with strings
            return f" {tablename}.{key} IN ({','.join(val_strs)})"
    raise Exception('Where clause options not recognized.')

