"""Script for inspecting MySQL database and testing injection functionalities"""

from pprint import pprint
import datetime
from typing import Dict, List

from src.crawler.crawler.utils.mysql_utils_ytvideos import (get_video_info_for_stats_spider)
from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import insert_records_from_dict, update_records_from_dict
from ytpa_utils.misc_utils import print_df_full
from ytpa_utils.time_utils import get_ts_now_str
from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_INFO
from src.crawler.crawler.constants import COL_VIDEO_ID, COL_USERNAME, COL_TIMESTAMP_ACCESSED, COL_TITLE

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']



def inject_toy_data():
    database = DB_VIDEOS_DATABASE

    # users table
    tablename = DB_VIDEOS_TABLES['users']
    d = dict(username='user1')
    insert_records_from_dict(database, tablename, d, DB_MYSQL_CONFIG)
    d = dict(username=['user2', 'user3'])
    insert_records_from_dict(database, tablename, d, DB_MYSQL_CONFIG)

    # video_meta table
    tablename = DB_VIDEOS_TABLES['meta']
    d1 = dict(
        video_id='SD544S3MO5',
        username='user3',
        title='blah',
        upload_date=get_ts_now_str(mode='date'),
        duration=48,
        keywords="['blah', 'meh']",
        description="This is a description.",
        thumbnail_url='url5.html',
        tags="['blahasdfasdfsdfa', 'sdasdfdsfsa']"
    )
    d2 = dict(
        video_id=['SD4J43MO5', '46HF4D8M'],
        username=['user1', 'user2'],
        title=['blah', 'hmmasdfsadf a a ammm'],
        upload_date=[get_ts_now_str(mode='date') for _ in range(2)],
        duration=[4833, 122],
        keywords=["['blah', 'meh']", "['ccccc', 'asdfg']"],
        description=["This is another description.", "And another description"],
        thumbnail_url=['url.html', 'url2.jpeg'],
        tags=["['blahasdfa', 'mehadsfsa']", "['cddddcc', 'asfdfdff']"]
    )
    keys_pre = [COL_VIDEO_ID, COL_USERNAME]
    d1_pre = {key: val for key, val in d1.items() if key in keys_pre}  # new record init
    d2_pre = {key: val for key, val in d2.items() if key in keys_pre}

    if 1:
        insert_records_from_dict(database, tablename, d1_pre, DB_MYSQL_CONFIG, keys=keys_pre)  # insert with just two keys
        insert_records_from_dict(database, tablename, d2_pre, DB_MYSQL_CONFIG, keys=keys_pre)
        update_records_from_dict(database, tablename, d1, DB_MYSQL_CONFIG)  # update with rest of info
        update_records_from_dict(database, tablename, d2, DB_MYSQL_CONFIG)

    # video_stats table
    tablename = DB_VIDEOS_TABLES['stats']
    d1 = dict(
        video_id='SD544S3MO5',
        timestamp_accessed=get_ts_now_str(mode='ms'),
        like_count=234234,
        view_count=48,
        subscriber_count=345345,
        comment_count=22,
        comment="This is a comment."
    )
    d2 = dict(
        video_id=['SD4J43MO5', '46HF4D8M'],
        timestamp_accessed=[get_ts_now_str(mode='ms') for _ in range(2)],
        like_count=[234, 65433],
        view_count=[48, 33],
        subscriber_count=[345345, 7676],
        comment_count=[22, 224],
        comment=["Blah blah comment.", "Another comment."]
    )
    if 1:
        insert_records_from_dict(database, tablename, d1, DB_MYSQL_CONFIG)
        insert_records_from_dict(database, tablename, d2, DB_MYSQL_CONFIG)


def inspect_videos_db(inject_data: bool = False):
    # insert rows to all tables using mysql utils
    if inject_data:
        inject_toy_data()

    # initialize MySQL engine
    engine = MySQLEngine(DB_MYSQL_CONFIG)

    # see table schemas and contents
    for _, tablename in DB_VIDEOS_TABLES.items():
        table = engine.describe_table(DB_VIDEOS_DATABASE, tablename)
        print('\n' + tablename)
        pprint(table)

        df = engine.select_records(DB_VIDEOS_DATABASE, f'SELECT * FROM {tablename}', mode='pandas', tablename=tablename)
        print('')
        print(len(df))
        print_df_full(df, row_lims=[0, 10])

    # inspect meta table
    if 0:
        tablename = DB_VIDEOS_TABLES['meta']
        df = engine.select_records(DB_VIDEOS_DATABASE, f"SELECT * FROM {tablename}",
                                   mode='pandas', tablename=tablename)
        print('')
        # print_df_full(df[COL_UPLOAD_DATE].drop_duplicates())
        # print_df_full(df[[COL_VIDEO_ID, COL_TITLE, COL_USERNAME, COL_UPLOAD_DATE]].drop_duplicates())
        print_df_full(df[df[COL_USERNAME] == 'CNN'])
        # print_df_full(df)

    # see video info for stats spider
    if 0:
        df = get_video_info_for_stats_spider()
        print_df_full(df)

    # try a join SQL query
    if 0:
        # join on meta and stats tables
        table_pseudoname_primary = 'stats'
        table_pseudoname_secondary = 'meta'
        cols = [
            f'{table_pseudoname_primary}.video_id',
            f'{table_pseudoname_primary}.timestamp_accessed',
            f'{table_pseudoname_primary}.comment',
            f'{table_pseudoname_secondary}.title',
            f'{table_pseudoname_secondary}.description'
        ]
        df = engine.select_records_with_join(
            DB_VIDEOS_DATABASE,
            DB_VIDEOS_TABLES['stats'],
            DB_VIDEOS_TABLES['meta'],
            f'{table_pseudoname_primary}.video_id = {table_pseudoname_secondary}.video_id',
            cols,
            table_pseudoname_primary=table_pseudoname_primary,
            table_pseudoname_secondary=table_pseudoname_secondary
        )
        print('')
        print_df_full(df, row_lims=[0, 100])

    # look at histogram of time coverage for videos
    if 1:
        tablename = DB_VIDEOS_TABLES['stats']
        df_gen = engine.select_records(DB_VIDEOS_DATABASE, f'SELECT * FROM {tablename}',
                                       mode='pandas', tablename=tablename, as_generator=True)

        dts: Dict[str, List[datetime]] = {} # video_id: min and max timestamp_accessed
        for df in df_gen:
        # while not (df := next(df_gen)).empty:
            gb = df[[COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED]].groupby([COL_VIDEO_ID])
            gb_min = gb.min()
            gb_max = gb.max()
            for key in gb_min.index:
                min_ = gb_min.loc[key, COL_TIMESTAMP_ACCESSED]
                max_ = gb_max.loc[key, COL_TIMESTAMP_ACCESSED]
                if key not in dts:
                    dts[key] = [min_, max_]
                else:
                    dts[key] = [min(dts[key][0], min_), max(dts[key][1], max_)]

        dt_diffs: Dict[str, float] = {key: (e[1] - e[0]).total_seconds() / 3600 for key, e in dts.items()}
        dt_diffs = {key: val / 24 for key, val in dt_diffs.items()} # days
        # dt_diffs = {key: val / 24 for key, val in dt_diffs.items() if 0 < val < 24 * 5} # days

        # print('')
        # print(len(dt_diffs))
        # pprint(dt_diffs)

        import matplotlib.pyplot as plt

        plt.hist(list(dt_diffs.values()), 50, (0, 5))
        plt.show()



if __name__ == '__main__':
    inspect_videos_db(inject_data=False)
