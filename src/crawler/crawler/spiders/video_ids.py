"""Spider for crawling video_id info off of users' "videos" pages"""

from typing import List
import re
import requests

import scrapy

from src.crawler.crawler.loggers import loggers
from ..utils.mysql_utils_ytvideos import get_user_video_page_urls_from_db
from ..config import RAWDATA_META_PUSH_ENDPOINT

LOGGER_CRAWLER = loggers.get('crawler')



class YouTubeLatestVideoIds(scrapy.Spider):
    """
    Once body of user's "videos" page is converted to a string, sections of the following form are available from which
    we can regex out the video id (i.e. url is www.youtube.com/watch?v=[video_id]):

    "url":"/watch?v=9YvIoAY7w50"

    The video id is included in many other places on the page's HTML for each video thumbnail, but this is the first in
    the section corresponding to each individual video. This returns the latest 20 or so video ids.
    """
    name = "yt-latest-video-ids"
    start_urls = get_user_video_page_urls_from_db()
    # start_urls = start_urls[:1]

    debug_info = True
    url_count = 0

    def parse(self, response):
        self.url_count += 1
        if self.debug_info and LOGGER_CRAWLER is not None:
            LOGGER_CRAWLER.debug('=' * 50)
            LOGGER_CRAWLER.debug(f'Processing URL {self.url_count}/{len(self.start_urls)}')
            LOGGER_CRAWLER.debug(response.url)

        ### Get info ###
        # get body as string and find video urls
        s: str = response.body.decode()
        regex = '"url":"\/watch\?v=([0-9A-Za-z]{11})"'
        video_ids: List[str] = list(set(re.findall(regex, s))) # returns portion in parentheses for substrings matching regex

        ### Update database ###
        username: str = response.url.split("/")[-2][1:]  # username portion starts with "@"
        d = dict(video_id=video_ids, username=[username] * len(video_ids))

        if self.debug_info and LOGGER_CRAWLER is not None:
            LOGGER_CRAWLER.debug(d)
            LOGGER_CRAWLER.debug('=' * 50)

        try:
            requests.post(RAWDATA_META_PUSH_ENDPOINT, json=d)
            if self.debug_info and LOGGER_CRAWLER is not None:
                LOGGER_CRAWLER.debug('Database injection was successful.')
        except Exception as e:
            if LOGGER_CRAWLER is not None:
                LOGGER_CRAWLER.exception('Exception in YouTubeLatestVideoIds.')
            else:
                print(e)
