"""Script for setting up MySQL database and tables for crawled video stats"""

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.paths import DB_VIDEO_SQL_FNAME
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


# initialize MySQL engine
engine = MySQLEngine(DB_CONFIG)

# drop database
if 0:
    res = input('Are you sure you want to delete this database (yes/no)?')
    if res == 'yes':
        engine.drop_db(DB_VIDEOS_DATABASE)

# create database
engine.create_db_from_sql_file(DB_VIDEO_SQL_FNAME)

# create tables
# engine.create_tables(DB_VIDEOS_DATABASE, queries_create_tables)
