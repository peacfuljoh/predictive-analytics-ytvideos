"""Script for inspecting MySQL database and tables for crawled video stats"""

from pprint import pprint

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


# initialize MySQL engine
engine = MySQLEngine(DB_CONFIG)


if 0:
    # insert rows
    username = 'Mr. Hat'
    video_ids = ['video1', 'video2', 'video55', 'new_video', 'www', 'vzzz']

    tablename = DB_VIDEOS_TABLENAMES['users']
    query = f"INSERT INTO {tablename} (username) VALUES ('{username}') ON DUPLICATE KEY UPDATE username=username"
    print(query)
    engine.insert_records_to_table(DB_VIDEOS_DATABASE, query)

    tablename = DB_VIDEOS_TABLENAMES['meta']
    query = f"INSERT INTO {tablename} (video_id, username) VALUES"
    query += ', '.join([f" ('{video_id}', '{username}')" for video_id in video_ids])
    query += " ON DUPLICATE KEY UPDATE username=username"
    print(query)
    engine.insert_records_to_table(DB_VIDEOS_DATABASE, query)


# see table schemas and contents
for _, tablename in DB_VIDEOS_TABLENAMES.items():
    table = engine.describe_table(DB_VIDEOS_DATABASE, tablename)
    print('\n' + tablename)
    pprint(table)

    table = engine.select_records(DB_VIDEOS_DATABASE, f'SELECT * FROM {tablename}', mode='pandas', tablename=tablename)
    print('')
    print(table)
