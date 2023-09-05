import re
import time
from typing import Optional, Union

import pandas as pd
import scrapy

from ..paths import USERNAMES_JSON_PATH
from ..utils.misc_utils import get_video_urls, get_video_stats_json_fname, save_json


def apply_regex(s: str,
                regex: str,
                dtype: Optional[str] = None)\
        -> Union[str, int]:
    """
    Apply regex to string to extract a string. Can handle further parsing if string is a number.
    Assumes regex is specified as string with embedded (.*?) to find substring.
    Handles commas separating sequences of digits (e.g. 12,473).
    """
    assert '(.*?)' in regex
    print(re.findall(regex, s))
    substring = re.findall(regex, s)[0] # find matching pattern, return portion at (.*?)
    if dtype == 'int':
        substring = substring.replace(',', '')
        if 'K' in substring:
            substring = substring[:-1].replace('.', '') + '00'
        num = int(substring) # remove commas and convert to int
        return num
    return substring


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

        # like count
        regex = '"defaultText":{"accessibility":{"accessibilityData":{"label":"(.*?) likes"}}'
        num_likes = apply_regex(s, regex, dtype='int')

        # view count
        regex = '"viewCount":{"videoViewCountRenderer":{"viewCount":{"simpleText":"(.*?) views"}'
        num_views = apply_regex(s, regex, dtype='int')

        # comment count
        regex = '"commentCount":{"simpleText":"(.*?)"},"contentRenderer"'
        num_comments = apply_regex(s, regex, dtype='int')

        # comment text (only captures most recent comment, no other comments are visible in response body)
        regex = '"teaserContent":{"simpleText":"(.*?)"},"trackingParams":"'
        comment = apply_regex(s, regex)

        # pack into dict
        df_row = self.df_urls.loc[self.df_urls['video_url'] == response.url]
        username = df_row['username'].iloc[0]
        video_id = df_row['video_id'].iloc[0]
        d = dict(
            like_count=num_likes,
            view_count=num_views,
            comment_count=num_comments,
            comment=comment,
            ts_access=int(time.time()),
            video_id=video_id,
            username=username
        )

        # save to JSON
        filename = get_video_stats_json_fname(username)
        save_json(filename, d, mode='a')
