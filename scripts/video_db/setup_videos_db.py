"""Script for setting up MySQL database and tables for crawled video stats"""

from typing import List

from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import insert_records_from_dict
from ytpa_utils.io_utils import load_json
from src.crawler.crawler.paths import DB_VIDEO_SQL_FNAME, USERNAMES_JSON_PATH
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

from inspect_videos_db import inspect_videos_db


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']



# initialize MySQL engine
engine = MySQLEngine(DB_CONFIG)

# drop database
if 0:
    res = input('Are you sure you want to delete this database (yes/no)?')
    if res == 'yes':
        engine.drop_db(DB_VIDEOS_DATABASE)

# create database
if 0:
    engine.create_db_from_sql_file(DB_VIDEO_SQL_FNAME)

# clear current usernames and insert from JSON file
if 0:
    tablename = DB_VIDEOS_TABLES['users']

    # clear
    if 1:
        query = f"DELETE FROM {tablename}" # without WHERE clause, will delete all entries in table
        if input(f'About to clear {tablename} table. Are you sure (yes/no)?') == 'yes':
            print(f'Deleting all entries in {tablename} table.')
            engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)

    # insert
    usernames: List[str] = load_json(USERNAMES_JSON_PATH)["usernames"]
    data = dict(username=usernames)
    insert_records_from_dict(DB_VIDEOS_DATABASE, tablename, data, DB_CONFIG)

# inspect result
inspect_videos_db()
