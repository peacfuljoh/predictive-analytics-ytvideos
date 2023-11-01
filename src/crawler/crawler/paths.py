
import os

from .config import DATA_ROOT, CRAWLER_ROOT


# files
USERNAMES_JSON_PATH = os.path.join(DATA_ROOT, 'usernames.json')
DB_VIDEO_SQL_FNAME = os.path.join(CRAWLER_ROOT, 'dbs', 'ytvideos.sql')
