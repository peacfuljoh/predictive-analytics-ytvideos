
from pprint import pprint

from src.crawler.crawler.utils.mysql_utils_ytvideos import get_video_info_for_stats_spider
from db_engines.mysql_engine import MySQLEngine
from ytpa_utils.misc_utils import print_df_full
from src.crawler.crawler.constants import COL_TIMESTAMP_FIRST_SEEN, COL_VIDEO_ID, COL_USERNAME, COL_TITLE
from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_INFO




def select_video_data():
    df = get_video_info_for_stats_spider(columns=[COL_USERNAME, COL_VIDEO_ID, COL_TITLE, COL_TIMESTAMP_FIRST_SEEN])

    print_df_full(df)

def read_sql_records():
    engine = MySQLEngine(DB_MYSQL_CONFIG)

    database = DB_INFO['DB_VIDEOS_DATABASE']
    tables = DB_INFO['DB_VIDEOS_TABLES']

    for tablename in tables.values():
        query = f'SELECT * FROM {tablename} LIMIT 5'
        print(query)
        schema = [tt[0] for tt in engine.describe_table(database, tablename)]
        print(schema)
        recs = engine.select_records(database, query)
        pprint(recs)


if __name__ == '__main__':
    # select_video_data()
    read_sql_records()