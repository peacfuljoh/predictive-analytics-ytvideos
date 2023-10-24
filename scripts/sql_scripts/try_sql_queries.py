
from src.crawler.crawler.utils.mysql_utils_ytvideos import get_video_info_for_stats_spider
from src.crawler.crawler.utils.misc_utils import print_df_full
from src.crawler.crawler.constants import COL_TIMESTAMP_FIRST_SEEN, COL_VIDEO_ID, COL_USERNAME, COL_TITLE


def select_video_data():
    df = get_video_info_for_stats_spider(columns=[COL_USERNAME, COL_VIDEO_ID, COL_TITLE, COL_TIMESTAMP_FIRST_SEEN])

    print_df_full(df)



if __name__ == '__main__':
    select_video_data()