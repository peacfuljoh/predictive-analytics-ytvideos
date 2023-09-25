
import os

from .config import DATA_ROOT, CRAWLER_ROOT


# directories
setup_dir = lambda path: os.makedirs(path, exist_ok=True)

VIDEOS_DATA_ROOT = os.path.join(DATA_ROOT, 'video_data')
VIDEO_IDS_DIR = os.path.join(VIDEOS_DATA_ROOT, "video_urls")
VIDEO_STATS_DIR = os.path.join(VIDEOS_DATA_ROOT, "video_stats")

# setup_dir(VIDEOS_DATA_ROOT)
# setup_dir(VIDEO_IDS_DIR)
# setup_dir(VIDEO_STATS_DIR)

# files
USERNAMES_JSON_PATH = os.path.join(DATA_ROOT, 'usernames.json')
DB_VIDEO_SQL_FNAME = os.path.join(CRAWLER_ROOT, 'dbs', 'ytvideos.sql')
