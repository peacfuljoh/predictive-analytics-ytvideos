import datetime
from typing import Union, Tuple

import requests
from requests import Response

from src.crawler.crawler.config import MONGO_INSERT_ONE_ENDPOINT
from src.crawler.crawler.constants import TIMESTAMP_FMT




REPL_STRS_TS_TO_MKEY = {'-': 'd', ' ': 's', ':': 'c', '.': 'p'}
REPL_STRS_MKEY_TO_TS = {val: key for key, val in REPL_STRS_TS_TO_MKEY.items()}



def post_one_record(database: str,
                    collection: str,
                    record: dict) \
        -> Response:
    """POST one record to the MongoDB database via the API endpoint."""
    post_data = {'database': database, 'collection': collection, 'record': record}
    res = requests.post(MONGO_INSERT_ONE_ENDPOINT, json=post_data)
    return res

def convert_ts_fmt_for_mongo_id(ts: Union[datetime, str],
                                replace_chars: bool = False) \
        -> Tuple[str, str]:
    # convert to string via format if not already there
    ts_str = ts if isinstance(ts, str) else ts.strftime(TIMESTAMP_FMT)

    # reformat for _id
    ts_str_id = ts_str
    if not replace_chars:
        for key in REPL_STRS_TS_TO_MKEY.keys():
            ts_str_id = ts_str_id.replace(key, '')
    else:
        for key, val in REPL_STRS_TS_TO_MKEY.items():
            ts_str_id = ts_str_id.replace(key, val)
    return ts_str, ts_str_id
