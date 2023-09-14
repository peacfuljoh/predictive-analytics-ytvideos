from typing import Union, List, Dict
import json
from pprint import pprint

import pandas as pd
import scrapy

from ..utils.misc_utils import convert_num_str_to_int, apply_regex, get_ts_now_str
from ..utils.db_mysql_utils import get_video_info_for_stats_spider, insert_records_from_dict, update_records_from_dict
from ..config import DB_INFO
from ..constants import VIDEO_URL_COL_NAME


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']



def extract_video_info_from_body(response,
                                 fmt: str = 'nosql') \
        -> Dict[str, Union[int, str, List[str]]]:
    """
    Get miscellaneous video info from video page body (decoded as string).

    'fmt' arg determines format for values of output dict. Default is 'nosql' which is JSON format. Can also specify
    'sql' which is SQL database compatible (all strings and ints).
    """
    assert fmt in ['nosql', 'sql']

    # get response body as a string
    s = response.body.decode()

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

    # subscriber count
    # regex = '"subscriberCountText":{"accessibility":{"accessibilityData":{"label":"(.*?) subscribers"}},"simpleText":"(.*?) subscribers"}'
    regex = 'subscribers"}},"simpleText":"(.*?) subscribers"}'
    d['subscriber_count'] = apply_regex(s, regex, dtype='int')

    # miscellaneous
    regex = '"videoDetails":{"videoId":"(.*?)","title":"(.*?)","lengthSeconds":"(.*?)","keywords":(.*?),' \
            '"channelId":"(.*?)","isOwnerViewing":(.*?),"shortDescription":"(.*?)","isCrawlable":(.*?),' \
            '"thumbnail":{"thumbnails":(.*?)},"allowRatings":(.*?),"viewCount":"(.*?)","author":"(.*?)"'

    _, title_, lengthSeconds_, keywords_, _, _, shortDescription_, _, thumbnail_, _, viewCount_, _ = \
        apply_regex(s, regex)[0]

    d['title']: str = title_

    d['duration']: int = convert_num_str_to_int(lengthSeconds_)

    d['keywords']: List[str] = json.loads(keywords_)

    shortDescription_: List[str] = shortDescription_.split('#') # (description, hashtags)
    d['description']: str = shortDescription_[0].replace('\\"', '"').replace('\\n', '')
    d['tags']: List[str] = shortDescription_[1:]

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
    df_videos: pd.DataFrame = get_video_info_for_stats_spider(columns=['username', 'video_id']) # 'video_url' appended
    start_urls = list(df_videos[VIDEO_URL_COL_NAME]) if df_videos is not None else []
    url_count = 0
    # start_urls = start_urls[:1] # for testing
    # start_urls = ["https://www.youtube.com/watch?v=8zM1L7wLTCw"]

    debug_info = True

    def parse(self, response):
        ### Get info ###
        self.url_count += 1
        if self.debug_info:
            print('=' * 50)
            print(f'Processing URL {self.url_count}/{len(self.start_urls)}')
            print(response.url)

        # get vid info from response body
        vid_info = extract_video_info_from_body(response, fmt='sql')
        df_row = self.df_videos.loc[self.df_videos[VIDEO_URL_COL_NAME] == response.url]
        vid_info['video_id'] = df_row['video_id'].iloc[0]
        vid_info['username'] = df_row['username'].iloc[0]
        vid_info['timestamp_accessed'] = get_ts_now_str(mode='ms')

        ### Insert to database ###
        if self.debug_info:
            pprint(vid_info)
            print('=' * 50)
        update_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLENAMES['meta'], vid_info)
        insert_records_from_dict(DB_VIDEOS_DATABASE, DB_VIDEOS_TABLENAMES['stats'], vid_info)
