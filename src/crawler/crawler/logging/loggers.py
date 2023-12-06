"""
Loggers for various parts of the YTPA system.
For an intro to logging in Python, see https://realpython.com/python-logging/.
"""

import os
import logging

from ytpa_utils.time_utils import get_ts_now_formatted

from ..config import LOG_DIR_PATH
from ..constants import TIMESTAMP_FMT_FPATH


# setup loggers if log files path is defined
loggers = {}
if LOG_DIR_PATH is not None:
    # log file info
    dt_str = get_ts_now_formatted(TIMESTAMP_FMT_FPATH)
    log_file_path = os.path.join(LOG_DIR_PATH, dt_str + '.log')

    # crawler
    for name_ in ['crawler']:
        logger = logging.getLogger(name_)

        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file_path)
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        c_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        loggers[name_] = logger
