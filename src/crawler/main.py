
import time

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from crawler.spiders.video_ids import YouTubeLatestVideoIds
from crawler.spiders.video_stats import YouTubeVideoStats

from crawler.config import AUTOCRAWL_CONFIG


meta_spider_name = YouTubeLatestVideoIds.name
stats_spider_name = YouTubeVideoStats.name


# run processes on a clock
t_start = time.time()
while 1:
    # crawl metadata
    process = CrawlerProcess(get_project_settings())
    process.crawl(meta_spider_name)
    process.start()

    # crawl stats
    process = CrawlerProcess(get_project_settings())
    process.crawl(stats_spider_name)
    process.start()

    # sleep until next crawl cycle
    t_now = time.time()
    t_elapsed = t_now - t_start
    if t_elapsed < AUTOCRAWL_CONFIG["REPEAT_INTVL"]:
        time.sleep(AUTOCRAWL_CONFIG["REPEAT_INTVL"] - t_elapsed)
    t_start += AUTOCRAWL_CONFIG["REPEAT_INTVL"]
