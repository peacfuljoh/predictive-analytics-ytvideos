
from typing import List
import re

import scrapy

from ..utils.db_mysql_utils import get_user_video_page_urls_from_db, insert_records_from_dict
from ..config import DB_INFO


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']



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

    def parse(self, response):
        ### Get info ###
        # get body as string and find video urls
        s: str = response.body.decode()
        regex = '"url":"\/watch\?v=([0-9A-Za-z]{11})"'
        video_ids: List[str] = list(set(re.findall(regex, s))) # returns portion in parentheses for substrings matching regex

        ### Update database ###
        tablename = DB_VIDEOS_TABLENAMES['meta']
        username: str = response.url.split("/")[-2][1:]  # username portion starts with "@"
        d = dict(
            video_id=video_ids,
            username=[username] * len(video_ids)
        )
        insert_records_from_dict(DB_VIDEOS_DATABASE, tablename, d, keys=list(d.keys()))
