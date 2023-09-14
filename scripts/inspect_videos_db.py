"""Script for inspecting MySQL database and testing injection functionalities"""

from pprint import pprint

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine, insert_records_from_dict, update_records_from_dict
from src.crawler.crawler.utils.misc_utils import get_ts_now_str, print_df_full
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


def inspect_videos_db(inject_data: bool = False):
    # initialize MySQL engine
    engine = MySQLEngine(DB_CONFIG)

    # insert rows to all tables using mysql utils
    if inject_data:
        database = DB_VIDEOS_DATABASE

        # users table
        tablename = DB_VIDEOS_TABLENAMES['users']
        d = dict(username='user1')
        insert_records_from_dict(database, tablename, d)
        d = dict(username=['user2', 'user3'])
        insert_records_from_dict(database, tablename, d)

        # video_meta table
        tablename = DB_VIDEOS_TABLENAMES['meta']
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
        keys_pre = ['video_id', 'username']
        d1_pre = {key: val for key, val in d1.items() if key in keys_pre} # new record init
        d2_pre = {key: val for key, val in d2.items() if key in keys_pre}
        insert_records_from_dict(database, tablename, d1_pre, keys=keys_pre) # insert with just two keys
        insert_records_from_dict(database, tablename, d2_pre, keys=keys_pre)
        update_records_from_dict(database, tablename, d1) # update with rest of info
        update_records_from_dict(database, tablename, d2)

        # video_stats table
        tablename = DB_VIDEOS_TABLENAMES['stats']
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
            insert_records_from_dict(database, tablename, d1)
            insert_records_from_dict(database, tablename, d2)


    # see table schemas and contents
    for _, tablename in DB_VIDEOS_TABLENAMES.items():
        table = engine.describe_table(DB_VIDEOS_DATABASE, tablename)
        print('\n' + tablename)
        pprint(table)

        df = engine.select_records(DB_VIDEOS_DATABASE, f'SELECT * FROM {tablename}', mode='pandas', tablename=tablename)
        print('')
        print_df_full(df)


    # try a join SQL query
    if 0:
        # join on meta and stats tables
        cols = ['s.video_id', 's.timestamp_accessed', 's.comment', 'm.title', 'm.description']
        df = engine.select_records_with_join(
            DB_VIDEOS_DATABASE,
            DB_VIDEOS_TABLENAMES['stats'],
            DB_VIDEOS_TABLENAMES['meta'],
            cols,
            's',
            'm',
            's.video_id = m.video_id'
        )
        print('')
        print_df_full(df)





if __name__ == '__main__':
    inspect_videos_db(inject_data=False)
