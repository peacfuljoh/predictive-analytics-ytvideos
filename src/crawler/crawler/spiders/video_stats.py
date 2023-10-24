"""Spider for crawling video stats off of individual video pages"""

from typing import Union, List, Dict
import json
from pprint import pprint

import pandas as pd
import scrapy

from ..utils.misc_utils import convert_num_str_to_int, apply_regex, get_ts_now_str, print_df_full
from ..utils.mysql_utils_ytvideos import get_video_info_for_stats_spider
from ..utils.mysql_engine import insert_records_from_dict, update_records_from_dict
from ..utils.mongodb_utils_ytvideos import fetch_url_and_save_image
from ..config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from ..constants import (COL_VIDEO_URL, MAX_LEN_DESCRIPTION, MAX_NUM_TAGS, MAX_LEN_TAG,
                         MAX_NUM_KEYWORDS, MAX_LEN_KEYWORD)


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']
DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE']
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']


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
    d['like_count'] = apply_regex(s, regex, dtype='int')

    # comment count
    regex = '"commentCount":{"simpleText":"(.*?)"},"contentRenderer"'
    d['comment_count'] = apply_regex(s, regex, dtype='int')

    # comment text (only captures most recent comment, no other comments are visible in response body)
    regex = '"teaserContent":{"simpleText":"(.*?)"},"trackingParams":"'
    d['comment'] = apply_regex(s, regex)

    # date posted
    regex = '"uploadDate":"(.*?)"'
    d['upload_date'] = apply_regex(s, regex)

    # subscriber count
    regex = 'subscribers"}},"simpleText":"(.*?) subscribers"}'
    d['subscriber_count'] = apply_regex(s, regex, dtype='int')

def parse_video_details(d: dict,
                        res: dict):
    """Parse VideoDetails dict extracted from response body and fill relevant fields in video info dict."""
    d['title']: str = res['title']

    d['duration']: int = convert_num_str_to_int(res['lengthSeconds'])

    if 'keywords' in res:
        d['keywords']: List[str] = [kw[:MAX_LEN_KEYWORD] for kw in res['keywords'][:MAX_NUM_KEYWORDS]]
    else:
        d['keywords']: List[str] = []

    shortDescription_: List[str] = res['shortDescription'].split('#')  # (description, hashtags)
    d['description']: str = shortDescription_[0].replace('\\"', '"').replace('\\n', '')
    d['description'] = d['description'][:MAX_LEN_DESCRIPTION]
    d['tags']: List[str] = [t[:MAX_LEN_TAG] for t in shortDescription_[1:MAX_NUM_TAGS + 1]]

    thumbnail_largest: Dict[str, Union[str, int]] = res['thumbnail']['thumbnails'][-1]  # list of dicts (get largest image)
    d['thumbnail_url']: str = thumbnail_largest['url']

    d['view_count']: int = convert_num_str_to_int(res['viewCount'])  # int

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
        d['keywords'] = json.dumps(d['keywords'])
        d['tags'] = json.dumps(d['tags'])

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
        self.df_videos: pd.DataFrame = get_video_info_for_stats_spider(columns=['username', 'video_id', 'title'])  # 'video_url' appended
        print_df_full(self.df_videos)
        # self.df_videos = df_videos[df_videos['title'].isnull()]
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
        vid_info['video_id'] = df_row['video_id'].iloc[0]
        vid_info['username'] = df_row['username'].iloc[0]
        ts_now_str = get_ts_now_str(mode='ms')
        vid_info['timestamp_accessed'] = ts_now_str
        vid_info['timestamp_first_seen'] = ts_now_str

        ### Insert stats info to MySQL database ###
        if self.debug_info:
            pprint(vid_info)
            print('=' * 50)

        update_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLES['meta'], vid_info, DB_CONFIG,
                                 another_condition='upload_date IS NULL') # to avoid overwriting timestamp_first_seen
        insert_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLES['stats'], vid_info, DB_CONFIG)

        ### Fetch and save thumbnail to MongoDB database ###
        key_ = 'thumbnail_url'
        if len(url := vid_info[key_]) > 0:
            try:
                fetch_url_and_save_image(DB_VIDEOS_NOSQL_DATABASE, DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'], DB_MONGO_CONFIG,
                                         vid_info['video_id'], url, verbose=True)
            except:
                print(f'Exception during MongoDB database injection for {key_}.')

        if self.debug_info:
            print('Database injection was successful.')
