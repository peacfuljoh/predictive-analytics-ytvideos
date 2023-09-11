
from typing import List
import re

import scrapy

from ..utils.db_mysql_utils import MySQLEngine, get_user_video_page_urls_from_db
from ..config import DB_CONFIG, DB_INFO, SCRAPY_CONFIG


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

    DOWNLOAD_DELAY = SCRAPY_CONFIG['DOWNLOAD_DELAY']

    def parse(self, response):
        ### Get info ###
        # get body as string and find video urls
        s: str = response.body.decode()
        regex = '"url":"\/watch\?v=([0-9A-Za-z]{11})"'
        video_ids: List[str] = list(set(re.findall(regex, s))) # returns portion in parentheses for substrings matching regex

        ### Update database ###
        tablename = DB_VIDEOS_TABLENAMES['meta']
        username: str = response.url.split("/")[-2][1:]  # username portion starts with "@"

        query = f"INSERT INTO {tablename} (video_id, username) VALUES"
        query += ', '.join([f" ('{video_id}', '{username}')" for video_id in video_ids])
        query += " ON DUPLICATE KEY UPDATE username=username"

        engine = MySQLEngine(DB_CONFIG)
        engine.insert_records_to_table(DB_VIDEOS_DATABASE, query)
