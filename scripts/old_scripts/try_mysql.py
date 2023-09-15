
from try_mysql_data import insert_reviewers_query, reviewers_records, queries_create_movie_ratings_tables, \
    insert_ratings_query, ratings_records, insert_movies_query

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine
from src.crawler.crawler.config import DB_CONFIG




engine = MySQLEngine(DB_CONFIG)



# create database
if 0:
    engine.create_db('test123')

# show databases on server
db_names = engine.get_db_names()
print(f'\n{db_names}')

# create tables in online_movie_rating database
engine.create_tables('online_movie_rating', queries_create_movie_ratings_tables)

# show table description
table = engine.describe_table('online_movie_rating', 'movies')
print(f'\n{table}')

# insert records
if 1:
    engine.insert_records_to_table('online_movie_rating', insert_movies_query)
    engine.insert_records_to_table('online_movie_rating', insert_reviewers_query, reviewers_records)
    engine.insert_records_to_table('online_movie_rating', insert_ratings_query, ratings_records)

# select records
print('')
tablenames = ['movies', 'reviewers', 'ratings']
for tablename in tablenames:
    select_movies_query = f"SELECT * FROM {tablename} LIMIT 50"
    records = engine.select_records('online_movie_rating', select_movies_query, mode='pandas', tablename=tablename)
    print(records)
