
import pandas as pd

from src.crawler.crawler.config import DB_INFO, DB_CONFIG
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine, update_records_from_dict



DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


engine = MySQLEngine(DB_CONFIG)


# define queries
queries = []

if 0:
    tablename = DB_VIDEOS_TABLENAMES["meta"]
    queries += [
        f'ALTER TABLE {tablename} MODIFY COLUMN title VARCHAR(200)',
        f'ALTER TABLE {tablename} MODIFY COLUMN keywords VARCHAR(1000)',
        f'ALTER TABLE {tablename} MODIFY COLUMN description VARCHAR(500)',
        f'ALTER TABLE {tablename} MODIFY COLUMN thumbnail_url VARCHAR(500)',
        f'ALTER TABLE {tablename} MODIFY COLUMN tags VARCHAR(500)'
    ]

if 0:
    tablename = DB_VIDEOS_TABLENAMES['stats']
    queries += [
        f'ALTER TABLE {tablename} MODIFY COLUMN comment_count MEDIUMINT UNSIGNED'
    ]

if 1:
    tablename = DB_VIDEOS_TABLENAMES['meta']
    query = f'SELECT video_id, upload_date FROM {tablename}'
    recs = engine.select_records(DB_VIDEOS_DATABASE, query, mode='pandas', cols=['video_id', 'upload_date'])
    ts_recs_dict = {
        'video_id': list(recs['video_id']),
        'timestamp_first_seen': [dt.strftime('%Y-%m-%d %H:%M:%S.%f') if dt is not None else None
                                 for dt in recs['upload_date']]
    }
    print(recs)
    print(ts_recs_dict)

    exit()

    query = f'ALTER TABLE {tablename} ADD timestamp_first_seen TIMESTAMP(3)'
    engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)
    update_records_from_dict(DB_VIDEOS_DATABASE, tablename, ts_recs_dict,
                             condition_keys=['video_id'], keys=['timestamp_first_seen'])

# run queries
if 0:
    for query in queries:
        engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)
