
from typing import Optional, List, Dict
import datetime
import os

from src.etl.prefeaturization_etl_utils import ETLRequestPrefeatures
from db_engines.mysql_engine import MySQLEngine
from db_engines.mysql_utils import insert_records_from_dict

from constants_tests import DB_VIDEOS_DATABASE, SCHEMA_SQL_ORIG_FNAME, SCHEMA_SQL_TEST_FNAME, DATA_SQL_TEST




def assert_lists_match(exp: list, res: list):
    """Assert that list returned by a function call matches the expected list."""
    assert isinstance(res, list)
    assert len(exp) == len(res)
    assert all([exp[i] == res[i] for i in range(len(exp))])


""" Setup for prefeaturization ETL """
def setup_mysql_test_db(db_config: dict,
                        database: str,
                        data: Optional[Dict[str, dict]] = None):
    """Create test database. Overwrite if it already exists."""
    assert database in [DB_VIDEOS_DATABASE] and 'yt' not in database  # be careful with real db!

    engine = MySQLEngine(db_config)

    if database in engine.get_db_names():
        engine.drop_db(database)

    assert os.path.exists(SCHEMA_SQL_TEST_FNAME)
    engine.create_db_from_sql_file(SCHEMA_SQL_TEST_FNAME)

    if data is not None:
        for tablename, data_ in data.items():
            insert_records_from_dict(database, tablename, data_, db_config)

def setup_test_schema_file(schema_orig_fname: str,
                           schema_test_fname: str):
    """Use non-test schema file to create test schema file"""
    assert 'ytvideos' in schema_orig_fname
    assert 'test' in schema_test_fname

    with open(schema_orig_fname, 'r') as fp:
        text = fp.read()
    text = text.replace('ytvideos', DB_VIDEOS_DATABASE)
    assert 'yt' not in text
    with open(schema_test_fname, 'w') as fp:
        fp.write(text)

def setup_for_prefeatures_tests(req: ETLRequestPrefeatures):
    """Set up and populate test tables with dummy values for testing prefeaturization ETL pipeline"""
    # create test schema file
    setup_test_schema_file(SCHEMA_SQL_ORIG_FNAME, SCHEMA_SQL_TEST_FNAME)

    # setup MySQL database and inject data
    db_ = req.get_db()
    db_config = db_['db_mysql_config']
    database = db_['db_info']['DB_VIDEOS_DATABASE']

    setup_mysql_test_db(db_config, database, DATA_SQL_TEST)

def verify_prefeatures_tests(req: ETLRequestPrefeatures):
    """Verify that data in prefeatures MongoDB database is as expected"""
    # TODO: implement this
    pass