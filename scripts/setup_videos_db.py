"""Script for setting up MySQL database and tables for crawled video stats"""

from pprint import pprint

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.db_info import DB_CONFIG, DB_VIDEOS_DATABASE, DB_VIDEOS_TABLENAMES


queries_create_tables = [
        """
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(50) PRIMARY KEY
            );
        """,
        """ 
            CREATE TABLE IF NOT EXISTS video_meta (
                video_id VARCHAR(20) PRIMARY KEY,
                username VARCHAR(50),
                title VARCHAR(100),
                upload_date TIMESTAMP,
                duration SMALLINT UNSIGNED,
                keywords VARCHAR(500),
                description VARCHAR(500),
                thumbnail_url VARCHAR(100),
                tags VARCHAR(500)
            );
        """,
        """
            CREATE TABLE IF NOT EXISTS video_stats (
                video_id VARCHAR(20),
                date_accessed TIMESTAMP,
                like_count INT,
                view_count INT,
                comment_count INT,
                comment VARCHAR(1000),
                FOREIGN KEY(video_id) REFERENCES video_meta(video_id),
                PRIMARY KEY(video_id, date_accessed)
            );
        """
]

# initialize MySQL engine
engine = MySQLEngine(DB_CONFIG)

# drop database
if 0:
    res = input('Are you sure you want to delete this database (yes/no)?')
    if res == 'yes':
        engine.drop_db(DB_VIDEOS_DATABASE)

# create database
engine.create_db(DB_VIDEOS_DATABASE)

# create tables
engine.create_tables(DB_VIDEOS_DATABASE, queries_create_tables)
