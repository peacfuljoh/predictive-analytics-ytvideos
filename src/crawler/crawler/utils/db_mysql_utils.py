"""Database utils for MySQL"""

from typing import Callable, Optional, List, Union, Dict

import pandas as pd

import mysql.connector



class MySQLEngine():
    def __init__(self,
                 db_config: Dict[str, str]):
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
                                records: Optional[str] = None):
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
