"""Run a cyclical crawl sequence to pull video metadata and statistics"""

from datetime import timedelta

from crawler.utils.spider_utils import run_crawler
from crawler.utils.misc_utils import TimeLock, get_dt_now
from crawler.config import AUTOCRAWL_CONFIG


dt_start = get_dt_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1) # start at next hour
time_lock = TimeLock(dt_start, AUTOCRAWL_CONFIG["REPEAT_INTVL"], verbose=True)

while 1:
    print('\n' * 10)
    time_lock.acquire()

    from crawler.spiders.video_ids import YouTubeLatestVideoIds
    from crawler.spiders.video_stats import YouTubeVideoStats

    meta_spider_name = YouTubeLatestVideoIds.name
    stats_spider_name = YouTubeVideoStats.name

    run_crawler(meta_spider_name)
    run_crawler(stats_spider_name)
