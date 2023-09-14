
import os
import json
import re
from typing import Union, List, Dict, Optional, Tuple
import datetime
import time

import pandas as pd


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

def get_ts_now_str(mode: str) -> str:
    """Get current timestamp"""
    def get_ts_now_formatted(fmt: str) -> str:
        return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)

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


if __name__ == '__main__':
    if 1:
        print(get_ts_now_str(mode='date'))
        print(get_ts_now_str(mode='s'))
        print(get_ts_now_str(mode='ms'))
        print(get_ts_now_str(mode='us'))
