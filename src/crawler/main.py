"""Run a cyclical crawl sequence to pull video metadata and statistics"""

from datetime import timedelta

from ytpa_utils.time_utils import TimeLock, get_dt_now
from crawler.utils.spider_utils import run_crawler
from crawler.config import AUTOCRAWL_CONFIG


dt_start = get_dt_now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1) # start at next hour
time_lock = TimeLock(dt_start, AUTOCRAWL_CONFIG["REPEAT_INTVL"], progress_dur=60 * 5, verbose=True)

while 1:
    try:
        print('\n' * 10)
        time_lock.acquire()

        from crawler.spiders.video_ids import YouTubeLatestVideoIds
        from crawler.spiders.video_stats import YouTubeVideoStats

        run_crawler(YouTubeLatestVideoIds.name)
        run_crawler(YouTubeVideoStats.name)
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        print(e)
        # print('\n\nCannot establish connection (internet may be down). '
        #       'Resetting TimeLock and waiting until next attempt.')

