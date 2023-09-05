
import os


DATA_ROOT = 'C:/Users/johtr/Desktop'
VIDEOS_DATA_ROOT = os.path.join(DATA_ROOT, 'video_data')
VIDEO_IDS_DIR = os.path.join(VIDEOS_DATA_ROOT, "video_urls")
VIDEO_STATS_DIR = os.path.join(VIDEOS_DATA_ROOT, "video_stats")

os.makedirs(VIDEO_IDS_DIR, exist_ok=True)
os.makedirs(VIDEO_STATS_DIR, exist_ok=True)

CRAWLER_ROOT = 'C:/Users/johtr/Desktop/crawler/src/crawler/crawler'

USERNAMES_JSON_PATH = os.path.join(CRAWLER_ROOT, 'usernames.json')
