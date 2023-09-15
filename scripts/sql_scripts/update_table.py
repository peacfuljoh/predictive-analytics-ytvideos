
from src.crawler.crawler.config import DB_INFO, DB_CONFIG
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine



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

if 1:
    tablename = DB_VIDEOS_TABLENAMES['stats']
    queries += [
        f'ALTER TABLE {tablename} MODIFY COLUMN comment_count MEDIUMINT UNSIGNED'
    ]


# run queries
for query in queries:
    engine.execute_pure_sql(DB_VIDEOS_DATABASE, query)
