"""MongoDB Engine or CRUD and other ops"""

from typing import Dict, Union, Optional, Callable, List, Tuple, Generator
import math

import pandas as pd

from pymongo import MongoClient
from pymongo.collection import Collection, ObjectId
from pymongo.errors import BulkWriteError

from ..utils.misc_utils import is_list_of_strings


FIND_MANY_GEN_MAX_COUNT = 1000



class MongoDBEngine():
    """Convenience class for interactions with a MongoDB database"""
    def __init__(self,
                 db_config: Dict[str, Union[str, int]],
                 database: Optional[str] = None,
                 collection: Optional[str] = None,
                 verbose: bool = False):
        self._db_config = db_config
        self._database = None
        self._collection = None
        self._verbose = verbose

        self.set_db_info(database=database, collection=collection)

        self._db_client = MongoClient(self._db_config['host'], self._db_config['port'])

    def __del__(self):
        self._db_client.close()

    def set_db_info(self,
                    database: Optional[str] = None,
                    collection: Optional[str] = None):
        """Set database and collection to be used in db op calls"""
        if database is not None:
            self._database = database
        if collection is not None:
            self._collection = collection


    ### connection config ###
    def get_db_config(self) -> Dict[str, str]:
        return self._db_config


    ## connection and exception handling ###
    def _query_wrapper(self, func: Callable):
        """Wrapper for exception handling during MongoDB queries"""
        try:
            return func()
        except Exception as e:
            print(e)


    ## DB inspection ##
    def get_all_databases(self) -> List[str]:
        """Get all databases"""
        return self._db_client.list_database_names()

    def get_all_collections(self, database: Optional[str] = None) -> Dict[str, List[str]]:
        """Get all collections by database or just those for a specified database"""
        if database is not None:
            databases = [database]
        else:
            databases = self.get_all_databases()
        return {database: self._db_client[database].list_collection_names() for database in databases}

    def get_ids(self) -> List[str]:
        """Get all IDs for a collection"""
        return [str(id) for id in self._get_collection().distinct('_id')]


    ## DB operations ##
    def _get_collection(self) -> Collection:
        """Get collection object for queries"""
        assert self._database is not None
        assert self._collection is not None
        return self._db_client[self._database][self._collection]

    def insert_one(self, record: dict):
        """Insert one record"""
        def func():
            cn = self._get_collection()

            if ('_id' in record) and (cn.find_one({"_id": record['_id']}) is not None):
                raise Exception(f'MongoDBEngine: A record with id {record["_id"]} already exists in collection '
                                f'{self._collection} of database {self._database}.')

            res = cn.insert_one(record)

            if self._verbose:
                print(f'MongoDBEngine: Inserted {1} record with id {res.inserted_id} in collection {self._collection} '
                      f'of database {self._database}.')

        return self._query_wrapper(func)

    def insert_many(self, records: List[dict]):
        """Insert many records"""
        def func():
            try:
                cn = self._get_collection()
                cn.insert_many(records, ordered=False)
            except BulkWriteError as e:
                if self._verbose:
                    writeErrors = e.details['writeErrors']
                    print(f"Failed to write {len(writeErrors)} out of {len(records)} records.")
        return self._query_wrapper(func)

    def update_one(self,
                   filter: dict,
                   update: dict,
                   upsert: bool = False):
        """Update a single record"""
        def func():
            cn = self._get_collection()
            cn.update_one(filter, update, upsert=upsert)
        return self._query_wrapper(func)

    def update_many(self,
                    filter: dict,
                    update: List[dict],
                    upsert: bool = False,
                    max_pipeline_len: Optional[int] = 1000):
        """Update records"""
        def func():
            cn = self._get_collection()
            num_pipelines = math.ceil(len(update) / max_pipeline_len)
            for i in range(num_pipelines):
                update_i = update[i * max_pipeline_len: (i + 1) * max_pipeline_len]
                if self._verbose:
                    print(f'Updating {len(update_i)} records.')
                cn.update_many(filter, update_i, upsert=upsert)
        return self._query_wrapper(func)

    def find_one_by_id(self, id: str) -> Optional[dict]:
        """Find a single record"""
        def func():
            cn = self._get_collection()
            return cn.find_one({"_id": id})
        return self._query_wrapper(func)

    def find_one(self,
                 filter: Optional[dict] = None,
                 projection: Optional[dict] = None):
        """Same as self.find_many_gen() but for a single record."""
        if filter is None:
            filter = {}
        if projection is None:
            projection = {}

        def func():
            cn = self._get_collection()
            cursor = cn.find(filter, projection, limit=1)
            return next(cursor, None)
        return self._query_wrapper(func)

    def find_many_by_ids(self,
                         ids: Optional[List[str]] = None,
                         limit: int = 0) \
            -> List[dict]:
        """Find many records"""
        def func():
            cn = self._get_collection()
            # args = {} if ids is None else {"_id": {"$in": [ObjectId(id_) for id_ in ids]}}
            filter = {} if ids is None else {"_id": {"$in": ids}}
            cursor = cn.find(filter, limit=limit)
            return [d for d in cursor]
        return self._query_wrapper(func)

    def find_many_gen(self,
                      filter: Optional[dict] = None,
                      projection: Optional[dict] = None) \
            -> Generator[pd.DataFrame, None, None]:
        """
        Generator of records given optional filter and projection arguments.

        Args:
        - 'filter' is a dict with options analogous to a WHERE clause in a SQL query.
            e.g. filter = dict(name={'$in': ['a', 'b']}).
        - 'projection' is a dict with fields to return, analogous to "SELECT item, status FROM ..."
          instead of "SELECT * FROM ..." in a SQL query:
            e.g. {"item": 1, "status": 1, "_id": 0} # this drops '_id' in the returned dict
        """
        if filter is None:
            filter = {}
        if projection is None:
            projection = {}

        def func():
            cn = self._get_collection()
            cursor = cn.find(filter, projection)
            while 1:
                recs: List[dict] = []
                for _ in range(FIND_MANY_GEN_MAX_COUNT):
                    rec_ = next(cursor, None)
                    if rec_ is None:
                        break
                    recs.append(rec_)
                yield pd.DataFrame.from_records(recs) if len(recs) > 0 else pd.DataFrame()

        return self._query_wrapper(func)

    def find_distinct_gen(self,
                          group: str,
                          filter: Optional[dict] = None) \
            -> Generator[pd.DataFrame, None, None]:
        """Find all distinct values of a given field"""
        def func():
            cn = self._get_collection()

            pipeline = []
            if filter is not None:
                pipeline += [filter]
            pipeline += [{"$group": {"_id": "$" + group}}]

            cursor = cn.aggregate(pipeline)

            while 1:
                recs: List[str] = []
                for _ in range(FIND_MANY_GEN_MAX_COUNT):
                    rec_ = next(cursor, None)
                    if rec_ is None:
                        break
                    recs.append(rec_['_id'])
                yield pd.DataFrame(recs, columns=[group]) if len(recs) > 0 else pd.DataFrame()

        return self._query_wrapper(func)

    def delete_many(self, ids: Optional[List[str]] = None):
        """Delete records by id"""
        assert isinstance(ids, list)
        def func():
            cn = self._get_collection()
            if isinstance(ids, list):
                filter = {"_id": {"$in": ids}}
            else:
                filter = {}
            cn.delete_many(filter)
        return self._query_wrapper(func)

    def delete_all_records(self, confirm_delete: str):
        """Delete all records in a collection"""
        if confirm_delete != 'yes':
            return
        def func():
            cn = self._get_collection()
            cn.delete_many({})
        return self._query_wrapper(func)


# def get_mongodb_records(database: str,
#                         collection: str,
#                         db_config: dict,
#                         ids: Optional[Union[str, List[str]]] = None,
#                         limit: int = 0) \
#         -> Union[dict, List[dict], None]:
#     """Get one or more MongoDB records from a single ID or list of IDs"""
#     assert ids is None or isinstance(ids, (str, list))
#
#     engine = MongoDBEngine(db_config, database=database, collection=collection)
#     if ids is None:
#         return engine.find_many(limit=limit)
#     if isinstance(ids, str):
#         return engine.find_one(ids)
#     if isinstance(ids, list):
#         return engine.find_many(ids=ids, limit=limit)

def get_mongodb_records_gen(database: str,
                            collection: str,
                            db_config: dict,
                            filter: Optional[dict] = None,
                            projection: Optional[dict] = None,
                            distinct: Optional[dict] = None) \
        -> Generator[pd.DataFrame, None, None]:
    """Prepare MongoDB feature generator with extract configuration."""
    using_filt = filter is not None
    using_proj = projection is not None
    using_dist = distinct is not None
    using_filt_or_proj = using_filt or using_proj
    assert not (using_filt and using_proj) # at most one can be specified of these two options
    assert not (using_filt_or_proj and using_dist) # using one or the other, but not both

    engine = MongoDBEngine(db_config, database=database, collection=collection)
    if distinct is not None:
        return engine.find_distinct_gen(distinct['group'], filter=distinct.get('filter'))
    else:
        filter_for_req: dict = {}
        for key, val in filter.items():
            if isinstance(val, str):
                filter_for_req[key] = val
            if is_list_of_strings(val):
                filter_for_req[key] = {'$in': val}
            else:
                raise NotImplementedError(f"Filter type not yet implemented: {key}: {val}.")

        return engine.find_many_gen(filter_for_req, projection=projection)