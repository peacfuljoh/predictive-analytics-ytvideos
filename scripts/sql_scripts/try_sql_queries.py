
from src.crawler.crawler.utils.mysql_utils_ytvideos import get_video_info_for_stats_spider
from src.crawler.crawler.utils.misc_utils import print_df_full


def select_video_data():
    df = get_video_info_for_stats_spider(columns=['username', 'video_id', 'title', 'timestamp_first_seen'])

    print_df_full(df)



if __name__ == '__main__':
    select_video_data()