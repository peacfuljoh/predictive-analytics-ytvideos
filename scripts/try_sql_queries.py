
from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine, get_video_info_for_stats_spider
from src.crawler.crawler.utils.misc_utils import print_df_full
from src.crawler.crawler.config import DB_CONFIG, DB_INFO


def select_video_data():
    # query = 'SELECT  FROM video_meta WHERE username = twosetviolin' # ORDER BY upload_date DESC LIMIT 11'
    # query = 'SELECT * FROM video_meta WHERE username = "twosetviolin" ORDER BY upload_date DESC'
    # query = 'SELECT video_id, username FROM video_meta WHERE username = "twosetviolin"'# AND upload_date IS NULL'
    #
    # engine = MySQLEngine(DB_CONFIG)
    #
    # df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query,
    #                            mode='pandas', cols=['video_id', 'username'])

    df = get_video_info_for_stats_spider(columns=['username', 'video_id'])

    print_df_full(df)



if __name__ == '__main__':
    select_video_data()