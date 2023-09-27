"""
Database utils for MySQL including functionality specific for the crawlers.
"""

from typing import Optional, List
import datetime

import pandas as pd
import numpy as np

from .misc_utils import make_video_urls, make_videos_page_urls_from_usernames, get_ts_now_str
from .mysql_engine import MySQLEngine
from ..config import DB_CONFIG, DB_INFO
from ..constants import (MOST_RECENT_VID_LIMIT, VIDEO_URL_COL_NAME, DB_KEY_TIMESTAMP_FIRST_SEEN,
                         VIDEO_STATS_CAPTURE_WINDOW_DAYS)




def get_usernames_from_db(usernames_desired: Optional[List[str]] = None) -> List[str]:
    """Get usernames from the users table"""
    engine = MySQLEngine(DB_CONFIG)

    tablename = DB_INFO["DB_VIDEOS_TABLENAMES"]["users"]
    query = f'SELECT * FROM {tablename}'
    usernames: List[tuple] = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query)
    usernames: List[str] = [e[0] for e in usernames]

    if usernames_desired is not None:
        set_usernames = set(usernames)
        set_usernames_desired = set(usernames_desired)
        assert len(set_usernames_desired - set_usernames) == 0 # ensure that desired usernames are all valid
        usernames = list(set_usernames.intersection(set_usernames_desired))

    return usernames

def get_video_info_for_stats_spider(usernames_desired: Optional[List[str]] = None,
                                    columns: Optional[List[str]] = None) \
        -> Optional[pd.DataFrame]:
    """
    Get video info from meta table for specified users.

    Options:
    - specify max number of records returned per user
    - make video urls and append to result
    """
    usernames: List[str] = get_usernames_from_db(usernames_desired=usernames_desired)

    assert len(usernames) > 0
    assert columns is None or (isinstance(columns, list) and len(columns) > 0)

    engine = MySQLEngine(DB_CONFIG)

    tablename: str = DB_INFO["DB_VIDEOS_TABLENAMES"]["meta"]

    dfs: List[pd.DataFrame] = []
    for username in usernames:
        # columns spec in query
        if columns is None:
            cols_str = '*'
        else:
            cols_str = f'{",".join(columns)}'

        # get DataFrame for non-null records, randomly sampling up to a max number of videos from the last X days
        ts_start = get_ts_now_str("ms", offset=datetime.timedelta(days=-VIDEO_STATS_CAPTURE_WINDOW_DAYS))
        query = (f'SELECT {cols_str} FROM {tablename} WHERE username = "{username}" AND upload_date IS NOT NULL'
                 f" AND {DB_KEY_TIMESTAMP_FIRST_SEEN} > '{ts_start}'")
                 #f' ORDER BY {DB_KEY_TIMESTAMP_FIRST_SEEN} DESC')

        if columns is None:
            df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', tablename=tablename)
        else:
            df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', cols=columns)
        if len(df) > MOST_RECENT_VID_LIMIT:
            df = df.iloc[np.random.permutation(len(df))[:MOST_RECENT_VID_LIMIT], :]

        # get DataFrame for null records
        query = (f'SELECT {cols_str} FROM {tablename} WHERE username = "{username}" AND upload_date IS NULL'
                 f' LIMIT {MOST_RECENT_VID_LIMIT}')

        if columns is None:
            df2 = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', tablename=tablename)
        else:
            df2 = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', cols=columns)

        # merge resulting DataFrames
        if df is None and df2 is None:
            continue
        if df is not None and df2 is None:
            pass
        if df is None and df2 is not None:
            df = df2
        if df is not None and df2 is not None:
            df = pd.concat((df, df2), axis=0)
        df = df.reset_index(drop=True)

        # add urls
        if not (df is None or df.empty):
            video_urls: List[str] = make_video_urls(list(df['video_id']))
            s_video_urls = pd.Series(video_urls, name=VIDEO_URL_COL_NAME)
            df = pd.concat((df, s_video_urls), axis=1)

        # save it
        dfs.append(df)

    if len(dfs) == 0:
        print('get_video_info_from_db_with_options() -> Could not find any records matching specified options.')
        print([usernames, columns])
        print(query)
        return None

    return pd.concat(dfs, axis=0, ignore_index=True)

def get_user_video_page_urls_from_db(usernames_desired: Optional[List[str]] = None) -> List[str]:
    """Get user video page URLs for all users listed in the database or a specified subset"""
    usernames: List[str] = get_usernames_from_db(usernames_desired=usernames_desired)
    urls: List[str] = make_videos_page_urls_from_usernames(usernames)
    return urls
