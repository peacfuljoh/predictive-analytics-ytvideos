
from typing import List
import re
import os

import scrapy

from src.utils import save_json


DATA_ROOT = 'C:/Users/johtr/Desktop'
VIDEOS_DATA_ROOT = os.path.join(DATA_ROOT, 'videos_data')
VIDEO_IDS_DIR = os.path.join(VIDEOS_DATA_ROOT, "video_urls")
os.makedirs(VIDEO_IDS_DIR, exist_ok=True)

USERNAMES = {
    'entertainment': [
        "StevenHe",
        "mrnigelng",
        "MrBeast",
        "howridiculous"
    ],
    'news': [
        "TheYoungTurks",
        "CNN",
        "FoxNews"
    ]
}
USERNAMES_FLAT = [elem for nested_list in USERNAMES.values() for elem in nested_list]
VIDEOS_USER_URLS = [f"https://www.youtube.com/@{username}/videos" for username in USERNAMES_FLAT]




class YouTubeSpider(scrapy.Spider):
    """
    Once body of user's "videos" page is converted to a string, sections of the following form are available from which
    we can regex out the video id (i.e. url is www.youtube.com/watch?v=[video_id]):

    "commandMetadata": { "webCommandMetadata": {"url":"/watch?v=9YvIoAY7w50","webPageType":"WEB_PAGE_TYPE_WATCH","rootVe":3832} }

    The video id is included in many other places on the page's HTML for each video thumbnail, but this is the first in
    the section corresponding to each individual video.
    """
    name = "ytvideos"
    start_urls = VIDEOS_USER_URLS

    def parse(self, response):
        # get body as string and find video urls
        s: str = response.body.decode()
        regex = '"url":"\/watch\?v=([0-9A-Za-z]{11})"'
        video_ids: List[str] = list(set(re.findall(regex, s))) # returns portion in parentheses for substrings matching regex

        # save video ids to JSON file
        username = response.url.split("/")[-2][1:]
        filename = os.path.join(VIDEO_IDS_DIR, f"{username}.json")
        save_json(filename, video_ids)
