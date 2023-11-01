
import pandas as pd

from src.crawler.crawler.config import DB_INFO, DB_MYSQL_CONFIG
from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import update_records_from_dict
from src.crawler.crawler.constants import COL_TIMESTAMP_FIRST_SEEN, COL_VIDEO_ID

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']


engine = MySQLEngine(DB_MYSQL_CONFIG)


# define queries
queries = []

if 0:
    tablename = DB_VIDEOS_TABLES["meta"]
    queries += [
        f'ALTER TABLE {tablename} MODIFY COLUMN title VARCHAR(200)',
        f'ALTER TABLE {tablename} MODIFY COLUMN keywords VARCHAR(1000)',
        f'ALTER TABLE {tablename} MODIFY COLUMN description VARCHAR(500)',
        f'ALTER TABLE {tablename} MODIFY COLUMN thumbnail_url VARCHAR(500)',
        f'ALTER TABLE {tablename} MODIFY COLUMN tags VARCHAR(500)'
    ]

if 0:
    tablename = DB_VIDEOS_TABLES['stats']
    queries += [
        f'ALTER TABLE {tablename} MODIFY COLUMN comment_count MEDIUMINT UNSIGNED'
    ]

if 1:
    tablename = DB_VIDEOS_TABLES['meta']
    query = f'SELECT video_id, upload_date FROM {tablename}'
    recs = engine.select_records(DB_VIDEOS_DATABASE, query, mode='pandas', cols=[COL_VIDEO_ID, COL_UPLOAD_DATE])
    ts_recs_dict = {
        COL_VIDEO_ID: list(recs[COL_VIDEO_ID]),
        COL_TIMESTAMP_FIRST_SEEN: [dt.strftime('%Y-%m-%d %H:%M:%S.%f') if dt is not None else None
                                 for dt in recs[COL_UPLOAD_DATE]]
    }
    print(recs)
    print(ts_recs_dict)

    exit()

    query = f'ALTER TABLE {tablename} ADD timestamp_first_seen TIMESTAMP(3)'
    engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)
    update_records_from_dict(DB_VIDEOS_DATABASE, tablename, ts_recs_dict,
                             condition_keys=[COL_VIDEO_ID], keys=[COL_TIMESTAMP_FIRST_SEEN])

# run queries
if 0:
    for query in queries:
        engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)
