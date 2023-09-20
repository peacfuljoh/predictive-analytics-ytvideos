"""Script for setting up MySQL database and tables for crawled video stats"""

from typing import List

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine, insert_records_from_dict
from src.crawler.crawler.utils.misc_utils import load_json
from src.crawler.crawler.paths import DB_VIDEO_SQL_FNAME, USERNAMES_JSON_PATH
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

from inspect_videos_db import inspect_videos_db


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']



# initialize MySQL engine
engine = MySQLEngine(DB_CONFIG)

# drop database
if 1:
    res = input('Are you sure you want to delete this database (yes/no)?')
    if res == 'yes':
        engine.drop_db(DB_VIDEOS_DATABASE)

# create database
engine.create_db_from_sql_file(DB_VIDEO_SQL_FNAME)

# insert usernames
if 1:
    tablename = DB_VIDEOS_TABLENAMES['users']
    usernames: List[str] = load_json(USERNAMES_JSON_PATH)["usernames"]
    data = dict(username=usernames)
    insert_records_from_dict(DB_VIDEOS_DATABASE, tablename, data)

# inspect result
inspect_videos_db()
