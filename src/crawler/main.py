"""Run the cyclical crawl sequence"""

import time

from crawler.spiders.video_ids import YouTubeLatestVideoIds
from crawler.spiders.video_stats import YouTubeVideoStats
from crawler.utils.spider_utils import run_crawler
from crawler.config import AUTOCRAWL_CONFIG


meta_spider_name = YouTubeLatestVideoIds.name
stats_spider_name = YouTubeVideoStats.name


t_start = time.time()
while 1:
    # crawl metadata
    run_crawler(meta_spider_name)

    # crawl stats
    run_crawler(stats_spider_name)

    # sleep until next crawl cycle
    t_now = time.time()
    t_elapsed = t_now - t_start
    if t_elapsed < AUTOCRAWL_CONFIG["REPEAT_INTVL"]:
        time.sleep(AUTOCRAWL_CONFIG["REPEAT_INTVL"] - t_elapsed)
    t_start += AUTOCRAWL_CONFIG["REPEAT_INTVL"]
