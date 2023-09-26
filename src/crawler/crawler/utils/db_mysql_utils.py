"""
Database utils for MySQL including basic CRUD operations and more
complex functionality specific for the crawlers.
"""

from typing import Callable, Optional, List, Union, Dict, Tuple, Generator

import pandas as pd

import mysql.connector

from .misc_utils import make_video_urls, make_videos_page_urls_from_usernames
from ..config import DB_CONFIG, DB_INFO
from ..constants import MOST_RECENT_VID_LIMIT, DB_KEY_UPLOAD_DATE, VIDEO_URL_COL_NAME, DB_KEY_TIMESTAMP_FIRST_SEEN



class MySQLEngine():
    """MySQL convenience class for CRUD and other operations on database records."""
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


    ### pure SQL ###
    def execute_pure_sql(self,
                         database: str,
                         query: str):
        def func(connection, cursor):
            cursor.execute(query)
            connection.commit()
        self._sql_query_wrapper(func, database=database)


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
            connection.commit()
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
            connection.commit()
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
                       tablename: Optional[str] = None,
                       cols: Optional[List[str]] = None,
                       as_generator: bool = False) \
            -> Union[Generator[pd.DataFrame, None, None], Generator[List[tuple], None, None], pd.DataFrame, List[tuple]]:
        """
        Retrieve records from a table.

        If mode == 'pandas', you must specify either the tablename (to infer all of that table's column names) or
        cols, which is the list of columns that the query will return. Either way, these column names will be used
        as the column names in the returned pandas dataframe.
        """
        assert mode in ['list', 'pandas']
        if mode == 'pandas':
            assert (tablename is None and cols is not None) or (tablename is not None and cols is None)
            if cols is None:
                cols = [e[0] for e in self.describe_table(database, tablename)]
        if mode == 'list':
            assert tablename is None

        if not as_generator:
            def func(connection, cursor):
                cursor.execute(query)
                records = cursor.fetchall() # if table empty, "1241 (21000): Operand should contain 1 column(s)"
                if mode == 'pandas':
                    return pd.DataFrame(records, columns=cols)
                return records

            return self._sql_query_wrapper(func, database=database)
        else:
            return self._select_records_gen(database, query, mode, cols=cols)

    def _select_records_gen(self,
                            database: str,
                            query: str,
                            mode: str = 'list',
                            cols: Optional[List[str]] = None):
        # sql_query_wrapper() doesn't work with yield...
        # Throws `mysql.connector.errors.ProgrammingError: 2055: Cursor is not connected`
        try:
            with self._get_connection(database=database) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    while (records := cursor.fetchmany(1000)) is not None:
                        if mode == 'pandas':
                            yield pd.DataFrame(records, columns=cols)
                        else:
                            yield records
        except mysql.connector.Error as e:
            print(e)

        # def func(connection, cursor):
        #     cursor.execute(query)
        #     while (records := cursor.fetchmany(100)) is not None:
        #         if mode == 'pandas':
        #             yield pd.DataFrame(records, columns=cols)
        #         else:
        #             yield records
        # return self._sql_query_wrapper(func, database=database)



    def select_records_with_join(self,
                                 database: str,
                                 tablename_primary: str,
                                 tablename_secondary: str,
                                 join_condition: str,
                                 cols_for_query: List[str],
                                 table_pseudoname_primary: Optional[str] = None,
                                 table_pseudoname_secondary: Optional[str] = None,
                                 where_clause: Optional[str] = None,
                                 limit: Optional[int] = None,
                                 cols_for_df: Optional[List[str]] = None,
                                 as_generator: bool = False) \
            -> Union[Generator[pd.DataFrame, None, None], pd.DataFrame]:
        """Select query on one table joined on second table"""
        if table_pseudoname_primary is None:
            table_pseudoname_primary = tablename_primary
        if table_pseudoname_secondary is None:
            table_pseudoname_secondary = tablename_secondary
        if cols_for_df is None:
            cols_for_df = cols_for_query
        query = (f"SELECT {', '.join(cols_for_query)} FROM {tablename_primary} as {table_pseudoname_primary} "
                 f"JOIN {tablename_secondary} as {table_pseudoname_secondary} ON {join_condition}")
        if where_clause is not None:
            query += f" WHERE {where_clause}"
        if limit is not None:
            query += f" LIMIT {limit}"

        return self.select_records(database, query, mode='pandas', cols=cols_for_df, as_generator=as_generator)



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

def get_video_info_for_stats_spider(usernames_desired: Optional[List[str]] = None,
                                    columns: Optional[List[str]] = None) \
        -> Optional[pd.DataFrame]:
    """
    Get video info from meta table for specified users.

    Options:
    - specify max number of records returned per user
    - make video urls and append to result
    """
    usernames: List[str] = get_usernames_from_db(usernames_desired=usernames_desired)

    assert len(usernames) > 0
    assert columns is None or (isinstance(columns, list) and len(columns) > 0)

    engine = MySQLEngine(DB_CONFIG)

    tablename: str = DB_INFO["DB_VIDEOS_TABLENAMES"]["meta"]

    dfs: List[pd.DataFrame] = []
    for username in usernames:
        # columns spec in query
        if columns is None:
            cols_str = '*'
        else:
            cols_str = f'{",".join(columns)}'

        # get DataFrame for non-null records
        query = (f'SELECT {cols_str} FROM {tablename} WHERE username = "{username}" AND upload_date IS NOT NULL '
                 f'ORDER BY {DB_KEY_TIMESTAMP_FIRST_SEEN} DESC LIMIT {MOST_RECENT_VID_LIMIT}')

        if columns is None:
            df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', tablename=tablename)
        else:
            df = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', cols=columns)

        # get DataFrame for null records
        query = (f'SELECT {cols_str} FROM {tablename} WHERE username = "{username}" AND upload_date IS NULL '
                 f'LIMIT {MOST_RECENT_VID_LIMIT}')

        if columns is None:
            df2 = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', tablename=tablename)
        else:
            df2 = engine.select_records(DB_INFO["DB_VIDEOS_DATABASE"], query, mode='pandas', cols=columns)

        # merge resulting DataFrames
        if df is None and df2 is None:
            continue
        if df is not None and df2 is None:
            pass
        if df is None and df2 is not None:
            df = df2
        if df is not None and df2 is not None:
            df = pd.concat((df, df2), axis=0)
        df = df.reset_index(drop=True)

        # add urls
        if not (df is None or df.empty):
            video_urls: List[str] = make_video_urls(list(df['video_id']))
            s_video_urls = pd.Series(video_urls, name=VIDEO_URL_COL_NAME)
            df = pd.concat((df, s_video_urls), axis=1)

        # save it
        dfs.append(df)

    if len(dfs) == 0:
        print('get_video_info_from_db_with_options() -> Could not find any records matching specified options.')
        print([usernames, columns])
        print(query)
        return None

    return pd.concat(dfs, axis=0, ignore_index=True)

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

def prep_keys_for_insert_or_update(database: str,
                                   tablename: str,
                                   data: dict,
                                   keys: Optional[List[str]] = None) \
        -> List[str]:
    """Validation and prep of keys for insert and update ops"""
    # if keys not specified, get all column names for this table
    if keys is None:
        keys = get_table_colnames(database, tablename)

    # key list is non-empty
    assert len(keys) > 0

    # keys is a subset of dict keys
    assert len(set(keys) - set(data.keys())) == 0

    # if multiple records, must have same number of records for all keys
    if isinstance(data[keys[0]], list):
        lens = [len(e) for e in data.values()]
        assert all([len_ == lens[0] for len_ in lens])

    return keys

def insert_records_from_dict(database: str,
                             tablename: str,
                             data: dict,
                             keys: Optional[List[str]] = None):
    """
    Insert all or a subset of the info from a dict to a database table. Subset is specified through 'keys' arg.

    Data dict could have individual entries or a list for each key. In the latter case, the number of entries must
    match for all keys.

    On duplicate key, do nothing.

    INSERT INTO table_name (column1, column2, column3, ...)
    VALUES (value1, value2, value3, ...)
    ON DUPLICATE KEY UPDATE column1=value1, ...;
    """
    keys = prep_keys_for_insert_or_update(database, tablename, data, keys=keys)

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
                             keys: Optional[List[str]] = None,
                             another_condition: Optional[str] = None):
    """
    Same as insert_records_from_dict() but applying an update operation.

    UPDATE table_name
    SET column1 = value1, column2 = value2, ...
    WHERE condition;

    'condition_keys' are the columns that are used in the WHERE clause (which records to update)
    'keys' are the columns to be updated
    """
    keys = prep_keys_for_insert_or_update(database, tablename, data, keys=keys)

    if condition_keys is None:
        condition_keys = get_table_primary_keys(database, tablename)
    assert len(condition_keys) > 0
    assert len(set(condition_keys) - set(data.keys())) == 0

    query = f"UPDATE {tablename} SET " + ', '.join([key + ' = %s' for key in keys])
    query += ' WHERE ' + ' AND '.join([key + ' = %s' for key in condition_keys])
    if another_condition is not None:
        query += ' AND ' + another_condition

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

