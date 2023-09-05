
import os
import json
from typing import Union, List, Dict, Optional, Tuple

import pandas as pd

from ..paths import VIDEO_IDS_DIR, VIDEO_STATS_DIR


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

def get_user_video_page_urls(usernames_json_path: str) -> List[str]:
    """Get user video page URLs from usernames JSON file"""
    usernames = load_json(usernames_json_path)
    usernames_flat = flatten_dict_of_lists(usernames)
    urls = make_videos_page_urls_from_usernames(usernames_flat)
    return urls

def get_video_ids_json_fname(username: str) -> str:
    return os.path.join(VIDEO_IDS_DIR, f"{username}.json")

def get_video_stats_json_fname(username: str) -> str:
    return os.path.join(VIDEO_STATS_DIR, f"{username}.json")

def get_video_urls(usernames_json_path: str,
                   categories: Optional[List[str]] = None,
                   usernames_desired: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Intended usage:
    - for users being tracked, get video urls for videos posted in the last x days (don't care about old videos)
    """
    # get usernames
    usernames = load_json(usernames_json_path)
    if categories is not None:
        assert all([cat in usernames for cat in categories])
        usernames = {key: val for key, val in usernames.items() if key in categories}
    usernames_flat: List[str] = flatten_dict_of_lists(usernames)
    if usernames_desired is not None:
        usernames_flat = list(set(usernames_flat).intersection(set(usernames_desired)))

    d: Dict[str, List[str]] = dict(username=[], video_id=[], video_url=[])
    for username in usernames_flat:
        video_ids, video_urls = get_video_urls_for_user(username)
        num_ids = len(video_ids)
        d['username'] += [username] * num_ids
        d['video_id'] += video_ids
        d['video_url'] += video_urls
    df = pd.DataFrame(d)
    return df

def get_video_urls_for_user(username: str) -> Tuple[List[str], List[str]]:
    """Get full video urls for a specified username"""
    video_ids_json_fname = get_video_ids_json_fname(username)
    video_ids: List[str] = load_json(video_ids_json_fname)
    urls = make_video_urls(video_ids)
    return video_ids, urls