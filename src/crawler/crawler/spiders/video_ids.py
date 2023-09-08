
from typing import List
import re

import scrapy

from ..utils.misc_utils import save_json, get_user_video_page_urls, get_video_ids_json_fname
from ..utils.db_mysql_utils import MySQLEngine
from ..paths import USERNAMES_JSON_PATH
from ..config import DB_CONFIG, DB_INFO

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


class YouTubeLatestVideoIds(scrapy.Spider):
    """
    Once body of user's "videos" page is converted to a string, sections of the following form are available from which
    we can regex out the video id (i.e. url is www.youtube.com/watch?v=[video_id]):

    "commandMetadata": { "webCommandMetadata": {"url":"/watch?v=9YvIoAY7w50","webPageType":"WEB_PAGE_TYPE_WATCH","rootVe":3832} }

    The video id is included in many other places on the page's HTML for each video thumbnail, but this is the first in
    the section corresponding to each individual video. This returns the latest 20 or so video ids.
    """
    name = "yt-latest-video-ids"
    start_urls = get_user_video_page_urls(USERNAMES_JSON_PATH)

    def parse(self, response):
        ### Get info ###
        # get body as string and find video urls
        s: str = response.body.decode()
        regex = '"url":"\/watch\?v=([0-9A-Za-z]{11})"'
        video_ids: List[str] = list(set(re.findall(regex, s))) # returns portion in parentheses for substrings matching regex

        # save video ids to JSON file
        username: str = response.url.split("/")[-2][1:] # username portion starts with "@"
        # filename = os.path.join(VIDEO_IDS_DIR, f"{username}.json")
        filename = get_video_ids_json_fname(username)
        save_json(filename, video_ids)

        ### Update database ###
        engine = MySQLEngine(DB_CONFIG)

        # update username table
        tablename = DB_VIDEOS_TABLENAMES['users']
        query = f"INSERT INTO {tablename} (username) VALUES ('{username}') ON DUPLICATE KEY UPDATE username=username"
        print(query)
        engine.insert_records_to_table(DB_VIDEOS_DATABASE, query)

        # update video ids
        tablename = DB_VIDEOS_TABLENAMES['meta']
        query = f"INSERT INTO {tablename} (video_id, username) VALUES"
        query += ', '.join([f" ('{video_id}', '{username}')" for video_id in video_ids])
        query += " ON DUPLICATE KEY UPDATE username=username"
        print(query)
        engine.insert_records_to_table(DB_VIDEOS_DATABASE, query)
