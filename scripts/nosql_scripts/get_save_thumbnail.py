
from pprint import pprint

from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import insert_records_from_dict, update_records_from_dict
from ytpa_utils.misc_utils import print_df_full
from ytpa_utils.misc_utils import fetch_data_at_url, convert_bytes_to_image
from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import COL_VIDEO_ID, COL_USERNAME, COL_TITLE, COL_THUMBNAIL_URL

from src.crawler.crawler.utils.mongodb_utils_ytvideos import fetch_url_and_save_image
from db_engines.mongodb_utils import get_mongodb_records

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']
DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE']
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']



engine = MySQLEngine(DB_MYSQL_CONFIG)

tablename = DB_VIDEOS_TABLES['meta']

cols = [COL_VIDEO_ID, COL_USERNAME, COL_TITLE, COL_THUMBNAIL_URL]
query = f'SELECT {",".join(cols)} FROM {tablename} WHERE thumbnail_url IS NOT NULL'# LIMIT 20'

df = engine.select_records(DB_VIDEOS_DATABASE, query, mode='pandas', cols=cols)

print_df_full(df)

if 0:
    for i, row in df.iterrows():
        fetch_url_and_save_image(DB_VIDEOS_NOSQL_DATABASE, DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'], DB_MONGO_CONFIG,
                                 row[COL_VIDEO_ID], row[COL_THUMBNAIL_URL], verbose=True, delay=5)

if 1:
    records = get_mongodb_records(DB_VIDEOS_NOSQL_DATABASE, DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'], DB_MONGO_CONFIG)
    # for i in range(len(records)):
    #     records[i]['img'] = records[i]['img'][::10000] # abbreviate it so it's not too long
    # pprint(records)
    print(f'num_records = {len(records)}')

if 0:
    records = get_mongodb_records(DB_VIDEOS_NOSQL_DATABASE, DB_VIDEOS_NOSQL_COLLECTIONS['thumbnails'], DB_MONGO_CONFIG)
    for rec in records:
        convert_bytes_to_image(rec['img']).show()
