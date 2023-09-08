import re
import time
from typing import Optional, Union, List, Dict
import json

import pandas as pd
import scrapy

from ..paths import USERNAMES_JSON_PATH
from ..utils.misc_utils import get_video_urls, get_video_stats_json_fname, save_json, convert_num_str_to_int


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
    print(res)
    if len(re.findall(substring_flag, regex)) > 1:
        return res
    substring = res[0] # find matching pattern, return portion at (.*?)
    if dtype == 'int':
        return convert_num_str_to_int(substring)
    return substring



def extract_video_info_from_body(s: str,
                                 fmt: str = 'nosql') \
        -> Dict[str, Union[int, str, List[str]]]:
    """Get miscellaneous video info"""
    d = {}

    # like count
    regex = '"defaultText":{"accessibility":{"accessibilityData":{"label":"(.*?) likes"}}'
    d['like_count'] = apply_regex(s, regex, dtype='int')

    # view count
    # regex = '"viewCount":{"videoViewCountRenderer":{"viewCount":{"simpleText":"(.*?) views"}'
    # d['view_count'] = apply_regex(s, regex, dtype='int')

    # comment count
    regex = '"commentCount":{"simpleText":"(.*?)"},"contentRenderer"'
    d['comment_count'] = apply_regex(s, regex, dtype='int')

    # comment text (only captures most recent comment, no other comments are visible in response body)
    regex = '"teaserContent":{"simpleText":"(.*?)"},"trackingParams":"'
    d['comment'] = apply_regex(s, regex)

    # date posted
    regex = '"uploadDate":"(.*?)"'
    d['upload_date'] = apply_regex(s, regex)

    # miscellaneous
    regex = '"videoDetails":{"videoId":"(.*?)","title":"(.*?)","lengthSeconds":"(.*?)","keywords":(.*?),' \
            '"channelId":"(.*?)","isOwnerViewing":(.*?),"shortDescription":"(.*?)","isCrawlable":(.*?),' \
            '"thumbnail":{"thumbnails":(.*?)},"allowRatings":(.*?),"viewCount":"(.*?)","author":"(.*?)"'

    video_id_, \
    title_, \
    lengthSeconds_, \
    keywords_, \
    channelId_, \
    isOwnerViewing_, \
    shortDescription_, \
    isCrawlable_, \
    thumbnail_, \
    allowRatings_, \
    viewCount_, \
    author_ = apply_regex(s, regex)[0]

    d['duration']: int = convert_num_str_to_int(lengthSeconds_)

    d['keywords']: List[str] = json.loads(keywords_)

    shortDescription_: List[str] = shortDescription_.split('\\n\\n')[:2]  # (description, hashtags)
    d['description']: str = shortDescription_[0]
    d['tags']: List[str] = shortDescription_[1].replace('#', '').split()

    thumbnail_largest: Dict[str, Union[str, int]] = json.loads(thumbnail_)[-1]  # list of dicts (get largest image)
    d['thumbnail_url']: str = thumbnail_largest['url']

    d['view_count']: int = convert_num_str_to_int(viewCount_)  # int

    # transform for output if necessary
    if fmt == 'sql':
        d['keywords'] = json.dumps(d['keywords'])
        d['tags'] = json.dumps(d['tags'])

    return d




class YouTubeVideoStats(scrapy.Spider):
    """
    Extract stats for specified videos.
    """
    name = "yt-video-stats"
    df_urls: pd.DataFrame = get_video_urls(USERNAMES_JSON_PATH)
    start_urls = list(df_urls.loc[:1, 'video_url'])

    def parse(self, response):
        # get response body as a string
        s = response.body.decode()

        # get duration, keywords, description, tags, thumbnail url, views
        vid_info = extract_video_info_from_body(s, fmt='sql')

        # pack into dict
        df_row = self.df_urls.loc[self.df_urls['video_url'] == response.url]
        username = df_row['username'].iloc[0]
        video_id = df_row['video_id'].iloc[0]
        meta_keys = ['title', 'upload_date', 'duration', 'keywords', 'description', 'thumbnail_url', 'tags']
        stats_keys = ['like_count', 'view_count', 'comment_count', 'comment']
        d_meta = {
            **dict(video_id=video_id, username=username),
            **{k:v for k, v in vid_info.items() if k in meta_keys}
        }
        d_stats = {
            **dict(video_id=video_id, date_accessed=int(time.time())),
            **{k:v for k, v in vid_info.items() if k in stats_keys}
        }

        # save to JSON
        filename = get_video_stats_json_fname(username)
        save_json(filename, d, mode='a')
