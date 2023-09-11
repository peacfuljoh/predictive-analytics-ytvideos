"""Script for inspecting MySQL database and tables for crawled video stats"""

from pprint import pprint

from src.crawler.crawler.utils.db_mysql_utils import MySQLEngine, insert_records_from_dict, update_records_from_dict
from src.crawler.crawler.utils.misc_utils import get_ts_now, print_df_full
from src.crawler.crawler.config import DB_CONFIG, DB_INFO

DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLENAMES = DB_INFO['DB_VIDEOS_TABLENAMES']


def inspect_videos_db():
    # initialize MySQL engine
    engine = MySQLEngine(DB_CONFIG)

    # insert rows to all tables using mysql utils
    if 1:
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
            upload_date=get_ts_now(as_string=True),
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
            upload_date=[get_ts_now(as_string=True), get_ts_now(as_string=True)],
            duration=[4833, 122],
            keywords=["['blah', 'meh']", "['ccccc', 'asdfg']"],
            description=["This is a description.", "And another description"],
            thumbnail_url=['url.html', 'url2.jpeg'],
            tags=["['blahasdfa', 'mehadsfsa']", "['cddddcc', 'asfdfdff']"]
        )
        keys_pre = ['video_id', 'username']
        d1_pre = {key: val for key, val in d1.items() if key in keys_pre}
        d2_pre = {key: val for key, val in d2.items() if key in keys_pre}
        insert_records_from_dict(database, tablename, d1_pre, keys=keys_pre) # insert with just two keys
        insert_records_from_dict(database, tablename, d2_pre, keys=keys_pre)
        update_records_from_dict(database, tablename, d1) # update with rest of info
        update_records_from_dict(database, tablename, d2)

        # video_stats table
        # TODO: implement this
        # TODO: replace hand-written insert (and other) queries with these utils


    # see table schemas and contents
    for _, tablename in DB_VIDEOS_TABLENAMES.items():
        table = engine.describe_table(DB_VIDEOS_DATABASE, tablename)
        print('\n' + tablename)
        pprint(table)

        df = engine.select_records(DB_VIDEOS_DATABASE, f'SELECT * FROM {tablename}', mode='pandas', tablename=tablename)
        print('')
        print_df_full(df)


if __name__ == '__main__':
    inspect_videos_db()

