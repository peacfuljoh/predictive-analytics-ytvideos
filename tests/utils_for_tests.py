
from typing import Optional, Dict, Union, List
import os

import pandas as pd

from src.etl.prefeaturization_etl_utils import ETLRequestPrefeatures
from db_engines.mysql_engine import MySQLEngine
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mysql_utils import insert_records_from_dict
from ytpa_utils.val_utils import is_list_of_strings
from constants_tests import (SCHEMA_SQL_ORIG_FNAME, SCHEMA_SQL_TEST_FNAME, DATA_SQL_TEST)
from src.crawler.crawler.config import (DB_VIDEOS_DATABASE_TEST, DB_VIDEOS_NOSQL_DATABASE_TEST,
                                        DB_MODELS_NOSQL_DATABASE_TEST, DB_FEATURES_NOSQL_DATABASE_TEST)



""" Helper methods """
def assert_lists_match(exp: list, res: list):
    """Assert that list returned by a function call matches the expected list."""
    assert isinstance(res, list)
    assert len(exp) == len(res)
    assert all([exp[i] == res[i] for i in range(len(exp))])




""" Database setup """
def setup_mysql_test_db(db_config: dict,
                        database: str,
                        data: Optional[Dict[str, Dict[str, list]]] = None):
    """
    Create test database. Overwrite if it already exists.

    nested keys in data: table, column
    """
    assert database in [DB_VIDEOS_DATABASE_TEST]
    assert 'yt' not in database  # be careful with real db!

    engine = MySQLEngine(db_config)

    if database in engine.get_db_names():
        engine.drop_db(database)

    assert os.path.exists(SCHEMA_SQL_TEST_FNAME)
    engine.create_db_from_sql_file(SCHEMA_SQL_TEST_FNAME)

    if data is not None:
        for tablename, data_ in data.items():
            insert_records_from_dict(database, tablename, data_, db_config)

def setup_mongo_test_dbs(db_config: dict,
                         databases: Union[str, List[str]],
                         data: Optional[Dict[str, Dict[str, Dict[str, list]]]] = None):
    """
    Create mongodb test database. Overwrite if it already exists.

    nested keys in data: database, collection, record key
    """
    assert isinstance(databases, str) or is_list_of_strings(databases)
    if isinstance(databases, str):
        databases = [databases]

    for database in databases:
        assert database in [DB_VIDEOS_NOSQL_DATABASE_TEST, DB_MODELS_NOSQL_DATABASE_TEST, DB_FEATURES_NOSQL_DATABASE_TEST]
        assert 'yt' not in database  # be careful with real db!

        engine = MongoDBEngine(db_config, database=database)

        engine.delete_all_records_in_database(database)

        if data is not None:
            for tablename, data_ in data.items():
                engine.insert_many(pd.DataFrame(data_).to_dict('records'))




""" Prefeaturization ETL """
def setup_test_schema_file(schema_orig_fname: str,
                           schema_test_fname: str):
    """Use non-test schema file to create test schema file"""
    assert 'ytvideos' in schema_orig_fname
    assert 'test' in schema_test_fname

    with open(schema_orig_fname, 'r') as fp:
        text = fp.read()
    text = text.replace('ytvideos', DB_VIDEOS_DATABASE_TEST)
    assert 'yt' not in text
    with open(schema_test_fname, 'w') as fp:
        fp.write(text)

def setup_for_prefeatures_tests(req: ETLRequestPrefeatures):
    """Set up and populate test tables with dummy values for testing prefeaturization ETL pipeline"""
    # create test schema file
    setup_test_schema_file(SCHEMA_SQL_ORIG_FNAME, SCHEMA_SQL_TEST_FNAME)

    db_ = req.get_db()

    # setup MySQL database and inject data
    db_config = db_['db_mysql_config']
    database_mysql = db_['db_info']['DB_VIDEOS_DATABASE']
    setup_mysql_test_db(db_config, database_mysql, DATA_SQL_TEST)

    # setup MongoDB database and inject data
    db_config = db_['db_mongo_config']
    databases_mongo = [db_['db_info'][db_name]
                       for db_name in ['DB_VIDEOS_NOSQL_DATABASE', 'DB_FEATURES_NOSQL_DATABASE']]
    setup_mongo_test_dbs(db_config, databases_mongo)

def verify_prefeatures_tests(req: ETLRequestPrefeatures):
    """Verify that data in prefeatures MongoDB database is as expected"""
    db_ = req.get_db()
    db_config = db_['db_mysql_config']
    database = db_['db_info']['DB_VIDEOS_DATABASE']

    engine = MySQLEngine(db_config)
    for tablename, data in DATA_SQL_TEST.items():
        df = engine.select_records(database, f'SELECT * FROM {tablename}', mode='pandas', tablename=tablename)
        # from ytpa_utils.misc_utils import print_df_full; print_df_full(df)
        assert df.equals(pd.DataFrame(data))



