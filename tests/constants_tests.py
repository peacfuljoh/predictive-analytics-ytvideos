
import datetime
import os

from src.crawler.crawler.constants import (COL_USERNAME, COL_VIDEO_ID, COL_TIMESTAMP_ACCESSED, COL_TITLE, COL_KEYWORDS,
                                           COL_TAGS, COL_COMMENT, COL_DESCRIPTION, COL_DURATION, COL_COMMENT_COUNT,
                                           COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT, COL_LIKE_COUNT, COL_UPLOAD_DATE,
                                           COL_TIMESTAMP_FIRST_SEEN, COL_THUMBNAIL_URL)
from src.crawler.crawler.config import REPO_ROOT



# sql schema filenames
SCHEMA_SQL_ORIG_FNAME = os.path.join(REPO_ROOT, 'src/crawler/crawler/dbs/ytvideos.sql')
SCHEMA_SQL_TEST_FNAME = os.path.join(REPO_ROOT, 'tests/schema_test.sql')

# raw data for testing
DATA_SQL_TEST = dict(
    users={
        COL_USERNAME: ['username1', 'username2']
    },
    video_meta={
        COL_VIDEO_ID: ['aaa', 'bbb', 'yyy'],
        COL_USERNAME: ['username1', 'username1', 'username2'],
        COL_TITLE: [
            'This is a title. It has "characters"! ',
            'So?\nYeah...it..does - not. ',
            'That’s $500.     blue?! ...  \n\n  '
        ],
        COL_UPLOAD_DATE: [
            datetime.date(2021, 7, 18),
            datetime.date(2021, 2, 23),
            datetime.date(2022, 11, 6)
        ],
        COL_TIMESTAMP_FIRST_SEEN: [
            datetime.datetime(2021, 12, 18, 0, 0),
            datetime.datetime(2022, 12, 14, 0, 0),
            datetime.datetime(2022, 12, 3, 0, 0)
        ],
        COL_DURATION: [30, 100, 95],
        COL_KEYWORDS: [
            '["231031__TA02IDF", "News", "Politics", "TYT", "The Young Turks", "Cenk Uygur", "Base", "Site 512"]',
            '["231030__TA01GroundOperationGaza", "News", "Politics", "TYT", "The Young Turks", "2024", "Israel", "Gaza", "Hamas"]',
            '["#Friends", "#Star", "#Matthew", "#Perry", "#Dead", "#ActorMatthewPerry"]'
        ],
        COL_DESCRIPTION: [
            '‘The Five’ Fox News Channel Live: http://www.foxnewsgo.com/\n\nFOX News (FNC) is all-encompassing ',
            'The Red NBC Blog: https://www.msnbc.com/reidoutblog\n\nMSNBC news, in-depth',
            'blah blah blah'
        ],
        COL_THUMBNAIL_URL: [
            'https://i.ytimg.com/vi/04GF6sV1iGs/maxresdefault.jpg',
            'https://i.ytimg.com/vi/04G43sV1iGs/maxresdefault.jpg',
            'https://i.ytimg.com/vi/04GF6sV2iGQ/maxresdefault.jpg'
        ],
        COL_TAGS: [
            '["Israel ", "HamasAttack ", "Gaza"]',
            '[]',
            '["SarahCooper ", "Author ", "Comedian"]'
        ]
    },
    video_stats={
        COL_VIDEO_ID: [
            'aaa', 'aaa', 'bbb', 'bbb', 'yyy', 'yyy'
        ],
        COL_TIMESTAMP_ACCESSED: [
            datetime.datetime(2021, 12, 9 + i, 0, 0) for i in range(6)
        ],
        COL_LIKE_COUNT: [35345 + i for i in range(6)],
        COL_VIEW_COUNT: [35345 + i for i in range(6)],
        COL_SUBSCRIBER_COUNT: [35345 + i for i in range(6)],
        COL_COMMENT_COUNT: [35345 + i for i in range(6)],
        COL_COMMENT: ['', 'cool ... ', 'huh?  ', 'it is a win', '55554  ', '  sdf ']
    }
)
