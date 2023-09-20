"""Spider for crawling video stats off of individual video pages"""

from typing import Union, List, Dict
import json
from pprint import pprint

import pandas as pd
import scrapy

from ..utils.misc_utils import convert_num_str_to_int, apply_regex, get_ts_now_str, fetch_data_at_url
from ..utils.db_mysql_utils import get_video_info_for_stats_spider, insert_records_from_dict, update_records_from_dict
from ..utils.db_mongo_utils import fetch_url_and_save_image
from ..config import DB_INFO
from ..constants import (VIDEO_URL_COL_NAME, MAX_LEN_DESCRIPTION, MAX_NUM_TAGS, MAX_LEN_TAG,
                         MAX_NUM_KEYWORDS, MAX_LEN_KEYWORD)


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']
DB_NOSQL_DATABASE = DB_INFO['DB_NOSQL_DATABASE']
DB_NOSQL_COLLECTION_NAMES = DB_INFO['DB_NOSQL_COLLECTION_NAMES']



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
    success = False

    regex = '"videoDetails":{"videoId":(.*?),"author":"(.*?)",'  # pull entire str-formatted dict
    res = apply_regex(s, regex)

    try:
        res = json.loads('{"videoId":' + res[0][0] + '}')
        parse_video_details(d, res)
        success = True
    except:
        print(f'\nCould not parse VideoDetails for {response.url}.\n')

    # no luck, write response body to file and set placeholder data
    if not success:
        video_id_ = response.url.split("=")[-1]
        with open(f'/home/nuc/crawler_data/{video_id_}.txt', 'w', encoding='utf-8') as fp:
            fp.write(s)

        d['title']: str = ''
        d['duration']: int = 0
        d['keywords']: List[str] = []
        d['description']: str = ''
        d['tags']: List[str] = []
        d['thumbnail_url']: str = ''
        d['view_count']: int = 0

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
    """
    assert fmt in ['nosql', 'sql']

    s = response.body.decode() # get response body as a string
    d = {}
    extract_individual_stats(d, s)
    extract_stats_from_video_details(d, s, fmt, response)

    return d




class YouTubeVideoStats(scrapy.Spider):
    """
    Extract stats for specified videos.
    """
    name = "yt-video-stats"
    df_videos: pd.DataFrame = get_video_info_for_stats_spider(columns=['username', 'video_id']) # 'video_url' appended
    start_urls = list(df_videos[VIDEO_URL_COL_NAME]) if df_videos is not None else []
    url_count = 0
    # start_urls = start_urls[:1] # for testing
    # start_urls = ["https://www.youtube.com/watch?v=TJ2ifmkGGus"]

    debug_info = True

    def parse(self, response):
        self.url_count += 1
        if self.debug_info:
            print('=' * 50)
            print(f'Processing URL {self.url_count}/{len(self.start_urls)}')
            print(response.url)

        ### Get stats info ###
        # get vid info from response body
        vid_info = extract_video_stats_from_response_body(response, fmt='sql')
        df_row = self.df_videos.loc[self.df_videos[VIDEO_URL_COL_NAME] == response.url]
        vid_info['video_id'] = df_row['video_id'].iloc[0]
        vid_info['username'] = df_row['username'].iloc[0]
        vid_info['timestamp_accessed'] = get_ts_now_str(mode='ms')

        ### Insert stats info to MySQL database ###
        if self.debug_info:
            pprint(vid_info)
            print('=' * 50)

        update_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLENAMES['meta'], vid_info)
        insert_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLENAMES['stats'], vid_info)

        ### Fetch and save thumbnail to MongoDB database ###
        if len(url := vid_info['thumbnail_url']) > 0:
            try:
                fetch_url_and_save_image(DB_NOSQL_DATABASE, DB_NOSQL_COLLECTION_NAMES['thumbnails'],
                                         vid_info['video_id'], url, verbose=True)
            except:
                pass

        if self.debug_info:
            print('Database injection was successful.')
