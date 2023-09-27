

import json
import re
from collections import OrderedDict
from typing import Tuple, Dict, List, Union, Optional, Generator
from pprint import pprint
import copy
import datetime

import numpy as np
import pandas as pd
from PIL import Image

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine, get_mongodb_records
from src.crawler.crawler.utils.mysql_engine import MySQLEngine
from src.crawler.crawler.utils.misc_utils import convert_bytes_to_image, is_datetime_formatted_str, df_generator_wrapper
from src.crawler.crawler.constants import STATS_ALL_COLS, META_ALL_COLS_NO_URL, STATS_NUMERICAL_COLS


DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE'] # NoSQL thumbnails
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


engine = MongoDBEngine(DB_MONGO_CONFIG, database=DB_FEATURES_NOSQL_DATABASE, collection=DB_FEATURES_NOSQL_COLLECTIONS)

data = engine.find_one('test')

a = 5