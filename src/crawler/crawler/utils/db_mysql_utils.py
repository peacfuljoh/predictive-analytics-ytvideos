"""Database utils for MySQL"""

from typing import Callable, Optional, List, Union, Dict, Tuple

import pandas as pd

import mysql.connector

from .misc_utils import make_video_urls, make_videos_page_urls_from_usernames
from ..config import DB_CONFIG, DB_INFO
from ..constants import MOST_RECENT_VID_LIMIT, DB_KEY_UPLOAD_DATE



class MySQLEngine():
    def __init__(self, db_config: Dict[str, str]):
        # members
        self._db_config = None

        # setup
        self.set_db_config(db_config)


    ### connection config ###
    def set_db_config(self, db_config: Dict[str, str]):
        self._db_config = db_config

    def get_db_config(self) -> Dict[str, str]:
        return self._db_config


    ### connection and exception handling ###
    def _get_connection(self, database: Optional[str] = None):
        """Establish connection with a MySQL database"""
        return mysql.connector.connect(
            host=self._db_config['host'],
            user=self._db_config['user'],
            password=self._db_config['password'],
            database=database
        )

    def _sql_query_wrapper(self,
                           func: Callable,
                           database: Optional[str] = None):
        """Wrapper for exception handling during MySQL queries"""
        try:
            with self._get_connection(database=database) as connection:
                with connection.cursor() as cursor:
                    return func(connection, cursor)
        except mysql.connector.Error as e:
            print(e)


    ### get database and table info ###
    def get_db_names(self) -> List[str]:
        """Get all names of existing databases"""
        def func(connection, cursor):
            cursor.execute("SHOW DATABASES")
            return [db_name[0] for db_name in cursor]
        return self._sql_query_wrapper(func)

    def describe_table(self,
                       database: str,
                       tablename: str) \
            -> List[tuple]:
        """Return schema of a table"""
        def func(connection, cursor):
            cursor.execute(f"DESCRIBE {tablename}")
            return cursor.fetchall()
        return self._sql_query_wrapper(func, database=database)


    ### operations on databases ###
    def create_db(self, db_name: str):
        """Create a database"""
        def func(connection, cursor):
            cursor.execute(f"CREATE DATABASE {db_name}")
        return self._sql_query_wrapper(func)

    def create_db_from_sql_file(self, filename: str):
        """Create a database from a .sql file with 'CREATE TABLE IF NOT EXISTS ...' statements"""
        assert '.sql' in filename
        def func(connection, cursor):
            with open(filename, 'r') as fd:
                sqlFile = fd.read()
            sqlCommands = sqlFile.split(';')
            for command in sqlCommands:
                if command.strip() != '':
                    cursor.execute(command)
            # with open(filename, 'r') as f:
            #     cursor.execute(f.read(), multi=True) # doesn't work...?
            connection.commit()
        return self._sql_query_wrapper(func)

    def drop_db(self, db_name: str):
        """Delete a database"""
        def func(connection, cursor):
            cursor.execute(f"DROP DATABASE {db_name}")
        return self._sql_query_wrapper(func)


    ### operations on tables ###
    def create_tables(self,
                      database: str,
                      queries: Union[str, List[str]]):
        """Create one or more tables in a specified database"""
        def func(connection, cursor):
            if isinstance(queries, str):
                cursor.execute(queries)
                connection.commit()
            else:
                for query in queries:
                    cursor.execute(query)
                    connection.commit()
        return self._sql_query_wrapper(func, database=database)

    def insert_records_to_table(self,
                                database: str,
                                query: str,
                                records: Optional[Union[str, List[tuple]]] = None):
        """Insert records into a table using a single query or a split query (instructions + data)"""
        def func(connection, cursor):
            if records is None:
                cursor.execute(query)
            else:
                cursor.executemany(query, records)
            connection.commit()
        return self._sql_query_wrapper(func, database=database)

    def select_records(self,
                       database: str,
                       query: str,
                       mode: str = 'list',
                       tablename: Optional[str] = None) \
            -> Union[pd.DataFrame, List[tuple]]:
        """Retrieve records from a table"""
        assert mode in ['list', 'pandas']
        if mode == 'pandas':
            assert tablename is not None
        if mode == 'list':
            assert tablename is None
        def func(connection, cursor):
            cursor.execute(query)
            records = cursor.fetchall()
            if mode == 'pandas':
                cols = [e[0] for e in self.describe_table(database, tablename)]
                return pd.DataFrame(records, columns=cols)
            return records
        return self._sql_query_wrapper(func, database=database)



### Helper functions ###
def get_usernames_from_db(usernames_desired: Optional[List[str]] = None) -> List[str]:
    """Get usernames from the users table"""
    engine = MySQLEngine(DB_CONFIG)

    tablename = DB_INFO["DB_VIDEOS_TABLENAMES"]["users"]
    query = f'SELECT * FROM {tablename}'
    usernames: List[tuple] = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query)
    usernames: List[str] = [e[0] for e in usernames]

    if usernames_desired is not None:
        set_usernames = set(usernames)
        set_usernames_desired = set(usernames_desired)
        assert len(set_usernames_desired - set_usernames) == 0 # ensure that desired usernames are all valid
        usernames = list(set_usernames.intersection(set_usernames_desired))

    return usernames

def get_video_info_from_db_with_options(usernames: List[str],
                                        num_limit: Optional[int] = None,
                                        columns: Optional[List[str]] = None,
                                        append_video_urls: bool = False) \
        -> pd.DataFrame:
    """Get video info for specified users"""
    engine = MySQLEngine(DB_CONFIG)

    tablename: str = DB_INFO["DB_VIDEOS_TABLENAMES"]["meta"]

    dfs: List[pd.DataFrame] = []
    for username in usernames:
        # get DataFrame with usernames and video_ids
        if columns is None:
            cols_str = '*'
        else:
            cols_str = f"({','.join(columns)})"
        query = f"SELECT {cols_str} FROM {tablename} WHERE username = '{username}' ORDER BY '{DB_KEY_UPLOAD_DATE}' DESC"
        if num_limit is not None:
            query += f" LIMIT {num_limit}"
        df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', tablename=tablename)

        # add video_urls
        if append_video_urls:
            video_urls: List[str] = make_video_urls(list(df['video_id']))
            df = df.concat((df, pd.Series(video_urls)), axis=1)

        # save it
        dfs.append(df)

    return pd.concat(dfs, axis=0)

def get_video_info_from_db(usernames_desired: Optional[List[str]] = None) -> pd.DataFrame:
    """For users being tracked, get recently-posted video urls"""
    usernames: List[str] = get_usernames_from_db(usernames_desired=usernames_desired)
    return get_video_info_from_db_with_options(
        usernames,
        num_limit=MOST_RECENT_VID_LIMIT,
        columns=['video_id', 'username'],
        append_video_urls=True
    )

def get_user_video_page_urls_from_db(usernames_desired: Optional[List[str]] = None) -> List[str]:
    """Get user video page URLs for all users listed in the database or a specified subset"""
    usernames: List[str] = get_usernames_from_db(usernames_desired=usernames_desired)
    urls: List[str] = make_videos_page_urls_from_usernames(usernames)
    return urls

def get_table_colnames(database: str,
                       tablename: str) \
        -> List[str]:
    """Get list of column names for a table in a database"""
    engine = MySQLEngine(DB_CONFIG)
    table_info = engine.describe_table(database, tablename)
    colnames = [tup[0] for tup in table_info]
    return colnames

def get_table_primary_keys(database: str,
                           tablename: str) \
        -> List[str]:
    """Get list of primary-key column names for a table in a database"""
    engine = MySQLEngine(DB_CONFIG)
    table_info = engine.describe_table(database, tablename)
    colnames = [tup[0] for tup in table_info if tup[3] == 'PRI']
    return colnames

def insert_records_from_dict(database: str,
                             tablename: str,
                             data: dict,
                             keys: Optional[List[str]] = None):
    """
    Insert all or a subset of the info from a dict to a database table.

    Data dict could have individual entries or a list for each key. In the latter case, the number of entries must
    match for all keys.

    On duplicate key, do nothing.

    INSERT INTO table_name (column1, column2, column3, ...)
    VALUES (value1, value2, value3, ...)
    ON DUPLICATE KEY UPDATE column1=value1, ...;
    """
    if keys is None:
        keys = get_table_colnames(database, tablename)
    assert len(keys) > 0
    assert len(set(keys) - set(data.keys())) == 0

    query = f"INSERT INTO {tablename} ({','.join(keys)}) VALUES (" + ','.join(['%s'] * len(keys)) + ")"
    query += f" ON DUPLICATE KEY UPDATE {keys[0]}={keys[0]}"

    if isinstance(data[keys[0]], list):
        records: List[tuple] = [tuple([data[key][i] for key in keys]) for i in range(len(data[keys[0]]))]
    else:
        records: List[tuple] = [tuple([data[key] for key in keys])]

    engine = MySQLEngine(DB_CONFIG)
    engine.insert_records_to_table(database, query, records)

def update_records_from_dict(database: str,
                             tablename: str,
                             data: dict,
                             condition_keys: Optional[List[str]] = None,
                             keys: Optional[List[str]] = None):
    """
    Same as insert_records_from_dict() but applying an update operation.

    UPDATE table_name
    SET column1 = value1, column2 = value2, ...
    WHERE condition;
    """
    if keys is None:
        keys = get_table_colnames(database, tablename)
    assert len(keys) > 0
    assert len(set(keys) - set(data.keys())) == 0

    if condition_keys is None:
        condition_keys = get_table_primary_keys(database, tablename)
    assert len(condition_keys) > 0
    assert len(set(condition_keys) - set(data.keys())) == 0

    query = f"UPDATE {tablename} SET " + ', '.join([key + ' = %s' for key in keys])
    query += ' WHERE ' + ', '.join([key + ' = %s' for key in condition_keys])

    if isinstance(data[keys[0]], list):
        records: List[tuple] = [
            tuple(
                [data[key][i] for key in keys] + [data[key][i] for key in condition_keys]
            )
            for i in range(len(data[keys[0]]))
        ]
    else:
        records: List[tuple] = [tuple([data[key] for key in keys] + [data[key] for key in condition_keys])]

    engine = MySQLEngine(DB_CONFIG)
    engine.insert_records_to_table(database, query, records)