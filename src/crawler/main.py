"""Run the cyclical crawl sequence"""

from datetime import timedelta

from crawler.spiders.video_ids import YouTubeLatestVideoIds
from crawler.spiders.video_stats import YouTubeVideoStats
from crawler.utils.spider_utils import run_crawler
from crawler.utils.misc_utils import TimeLock, get_dt_now
from crawler.config import AUTOCRAWL_CONFIG


meta_spider_name = YouTubeLatestVideoIds.name
stats_spider_name = YouTubeVideoStats.name


dt_start = get_dt_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1) # start at next hour
time_lock = TimeLock(dt_start, AUTOCRAWL_CONFIG["REPEAT_INTVL"], verbose=True)

while 1:
    time_lock.acquire()
    run_crawler(meta_spider_name)
    run_crawler(stats_spider_name)
