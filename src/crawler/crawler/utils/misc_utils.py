
import os
import json
import re
from typing import Union, List, Dict, Optional, Tuple, Any
import datetime
import time
from datetime import timedelta
import copy
from PIL import Image
import requests
import io

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

def get_dt_now() -> datetime.datetime:
    """Get current datetime"""
    return datetime.datetime.fromtimestamp(time.time())

class TimeLock():
    """
    Utility for managing timed events. TimeLock forces a waiting period until the next point in time an integer factor
    of intervals from an initial starting time. This allows for triggering events at strict intervals, where if one
    of those intervals is missed, it is skipped.
    """
    def __init__(self,
                 dt_start: datetime.datetime,
                 interval: int,
                 verbose: bool = False):
        assert (dt_start - get_dt_now()).total_seconds() > 0

        self._dt_target = dt_start
        self._interval = interval
        self._verbose = verbose

    def _wait_until_target(self):
        """Wait until current time catches up to target time"""
        t_wait = (self._dt_target - get_dt_now()).total_seconds()
        if t_wait > 0:
            if self._verbose:
                print(f'TimeLock: Waiting {t_wait} seconds until {self._dt_target}.')
            time.sleep(t_wait)

    def _advance_target(self):
        """Advance target time beyond the current time"""
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
    """Fetch data at specified URL"""
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
    if not isinstance(s, str):
        return False
    try:
        datetime.datetime.strptime(s, fmt)
        return True
    except Exception as e:
        return False
