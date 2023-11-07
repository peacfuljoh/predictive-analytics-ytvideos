
import os

os.environ['RUN_TESTS'] = 'yes' # always set to  'yes' for tests!

from src.crawler.crawler.utils.mysql_utils_ytvideos import make_videos_page_urls_from_usernames
from utils_for_tests import assert_lists_match



def test_make_videos_page_urls_from_username():
    in_ = ['bb', 'cc', '583']
    exp = [
        "https://www.youtube.com/@bb/videos",
        "https://www.youtube.com/@cc/videos",
        "https://www.youtube.com/@583/videos"
    ]

    res = make_videos_page_urls_from_usernames(in_)

    assert_lists_match(exp, res)

