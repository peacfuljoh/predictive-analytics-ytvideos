"""Spider for crawling video stats off of individual video pages"""

from typing import Union, List, Dict
import json
from pprint import pprint
import requests

import pandas as pd
import scrapy

from ytpa_utils.misc_utils import convert_num_str_to_int, apply_regex, print_df_full
from ytpa_utils.time_utils import get_ts_now_str
from ..utils.mysql_utils_ytvideos import get_video_info_for_stats_spider
from ..config import RAWDATA_STATS_PUSH_ENDPOINT
from ..constants import (COL_VIDEO_URL, MAX_LEN_DESCRIPTION, MAX_NUM_TAGS, MAX_LEN_TAG,
                         MAX_NUM_KEYWORDS, MAX_LEN_KEYWORD, COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN,
                         COL_DURATION, COL_VIDEO_ID, COL_USERNAME, COL_LIKE_COUNT, COL_COMMENT_COUNT,
                         COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT, COL_TIMESTAMP_ACCESSED, COL_COMMENT, COL_TITLE,
                         COL_KEYWORDS, COL_DESCRIPTION, COL_TAGS, COL_THUMBNAIL_URL)



def handle_extraction_failure(s: str, response):
    video_id_ = response.url.split("=")[-1]
    with open(f'/home/nuc/crawler_data/{video_id_}.txt', 'w', encoding='utf-8') as fp:
        fp.write(s)
    raise Exception(f'\n\n!!!!! Could not parse response body for {response.url}. !!!!!\n\n')

def extract_individual_stats(d: dict,
                             s: str):
    """Extract individual stats with their own regexes."""
    # like count
    regex = '"defaultText":{"accessibility":{"accessibilityData":{"label":"(.*?) likes"}}'
    d[COL_LIKE_COUNT] = apply_regex(s, regex, dtype='int')

    # comment count
    regex = '"commentCount":{"simpleText":"(.*?)"},"contentRenderer"'
    d[COL_COMMENT_COUNT] = apply_regex(s, regex, dtype='int')

    # comment text (only captures most recent comment, no other comments are visible in response body)
    regex = '"teaserContent":{"simpleText":"(.*?)"},"trackingParams":"'
    d[COL_COMMENT] = apply_regex(s, regex)

    # date posted
    regex = '"uploadDate":"(.*?)"'
    d[COL_UPLOAD_DATE] = apply_regex(s, regex)

    # subscriber count
    regex = 'subscribers"}},"simpleText":"(.*?) subscribers"}'
    d[COL_SUBSCRIBER_COUNT] = apply_regex(s, regex, dtype='int')

def parse_video_details(d: dict,
                        res: dict):
    """Parse VideoDetails dict extracted from response body and fill relevant fields in video info dict."""
    d[COL_TITLE]: str = res['title']

    d[COL_DURATION]: int = convert_num_str_to_int(res['lengthSeconds'])

    if 'keywords' in res:
        d[COL_KEYWORDS]: List[str] = [kw[:MAX_LEN_KEYWORD] for kw in res['keywords'][:MAX_NUM_KEYWORDS]]
    else:
        d[COL_KEYWORDS]: List[str] = []

    shortDescription_: List[str] = res['shortDescription'].split('#')  # (description, hashtags)
    d[COL_DESCRIPTION]: str = shortDescription_[0].replace('\\"', '"').replace('\\n', '')
    d[COL_DESCRIPTION] = d[COL_DESCRIPTION][:MAX_LEN_DESCRIPTION]
    d[COL_TAGS]: List[str] = [t[:MAX_LEN_TAG] for t in shortDescription_[1:MAX_NUM_TAGS + 1]]

    thumbnail_largest: Dict[str, Union[str, int]] = res['thumbnail']['thumbnails'][-1]  # list of dicts (get largest image)
    d[COL_THUMBNAIL_URL]: str = thumbnail_largest['url']

    d[COL_VIEW_COUNT]: int = convert_num_str_to_int(res['viewCount'])  # int

def extract_stats_from_video_details(d: dict,
                                     s: str,
                                     fmt: str,
                                     response):
    """Extract miscellaneous information from response body."""
    regex = '"videoDetails":{"videoId":(.*?),"author":"(.*?)",'  # pull entire str-formatted dict
    res = apply_regex(s, regex)
    res = json.loads('{"videoId":' + res[0][0] + '}')
    parse_video_details(d, res)

    # transform for output if necessary
    if fmt == 'sql':
        d[COL_KEYWORDS] = json.dumps(d[COL_KEYWORDS])
        d[COL_TAGS] = json.dumps(d[COL_TAGS])

def extract_video_stats_from_response_body(response,
                                           fmt: str = 'nosql') \
        -> Dict[str, Union[int, str, List[str]]]:
    """
    Get all relevant video stats from video page body (decoded as string).

    'fmt' arg determines format for values of output dict. Default is 'nosql' which is JSON format. Can also specify
    'sql' which is SQL database compatible (all strings and ints).

    To save from scrapy shell to file:
        s = response.body.decode(); f = open('/home/nuc/Desktop/temp/webpage.txt', 'w', encoding='utf-8'); f.write(s)
    """
    assert fmt in ['nosql', 'sql']

    s = response.body.decode() # get response body as a string
    d = {}
    try:
        extract_individual_stats(d, s)
        extract_stats_from_video_details(d, s, fmt, response)
    except Exception:
        handle_extraction_failure(s, response)

    return d




class YouTubeVideoStats(scrapy.Spider):
    """
    Extract stats for specified videos.
    """
    name = "yt-video-stats"

    debug_info = True

    def __init__(self, *args, **kwargs):
        super(YouTubeVideoStats, self).__init__(*args, **kwargs)
        self.reset_start_urls()

    def reset_start_urls(self):
        self.df_videos: pd.DataFrame = get_video_info_for_stats_spider(columns=[COL_USERNAME, COL_VIDEO_ID, COL_TITLE])  # 'video_url' appended
        print_df_full(self.df_videos)
        # self.df_videos = df_videos[df_videos[COL_TITLE].isnull()]
        self.start_urls = list(self.df_videos[COL_VIDEO_URL]) if self.df_videos is not None else []
        self.url_count = 0
        # self.start_urls = start_urls[:1] # for testing

    def parse(self, response):
        self.url_count += 1
        if self.debug_info:
            print('=' * 50)
            print(f'Processing URL {self.url_count}/{len(self.start_urls)}')
            print(response.url)

        ### Get stats info ###
        # get vid info from response body
        vid_info = extract_video_stats_from_response_body(response, fmt='sql')
        df_row = self.df_videos.loc[self.df_videos[COL_VIDEO_URL] == response.url]
        vid_info[COL_VIDEO_ID] = df_row[COL_VIDEO_ID].iloc[0]
        vid_info[COL_USERNAME] = df_row[COL_USERNAME].iloc[0]
        ts_now_str = get_ts_now_str(mode='ms')
        vid_info[COL_TIMESTAMP_ACCESSED] = ts_now_str
        vid_info[COL_TIMESTAMP_FIRST_SEEN] = ts_now_str

        ### Insert stats info to MySQL database ###
        if self.debug_info:
            pprint(vid_info)
            print('=' * 50)

        try:
            requests.post(RAWDATA_STATS_PUSH_ENDPOINT, json=vid_info)
            if self.debug_info:
                print('Database injection was successful.')
        except Exception as e:
            print(e)

