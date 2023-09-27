"""MongoDB Engine or CRUD and other ops"""

from typing import Dict, Union, Optional, Callable, List, Tuple
import math

from pymongo import MongoClient
from pymongo.collection import Collection


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
    # TODO: implement get_databases()
    # TODO: implement get_collections()


    ## DB operations ##
    def _get_collection(self) -> Collection:
        assert self._database is not None
        assert self._collection is not None
        return self._db_client[self._database][self._collection]

    def insert_one(self, record: dict):
        def func():
            assert '_id' in record
            cn = self._get_collection()
            if cn.find_one({"_id": record['_id']}) is not None:
                raise Exception(f'MongoDBEngine: A record with id {record["_id"]} already exists in collection '
                                f'{self._collection} of database {self._database}.')
            res = cn.insert_one(record)
            if self._verbose:
                print(f'MongoDBEngine: Inserted {1} record with id {res.inserted_id} in collection {self._collection} '
                      f'of database {self._database}.')
        return self._query_wrapper(func)

    def update_records(self,
                       filter: dict,
                       update: List[dict],
                       upsert: bool = False):
        MAX_PIPELINE_LEN = 1000
        def func():
            cn = self._get_collection()
            num_pipelines = math.ceil(len(update) / MAX_PIPELINE_LEN)
            for i in range(num_pipelines):
                cn.update_many(filter, update[i * MAX_PIPELINE_LEN: (i + 1) * MAX_PIPELINE_LEN], upsert=upsert)
        return self._query_wrapper(func)

    def find_one(self, id: str) -> Optional[dict]:
        def func():
            cn = self._get_collection()
            return cn.find_one({"_id": id})
        return self._query_wrapper(func)

    def find_many(self,
                  ids: Optional[List[str]] = None,
                  limit: int = 0) \
            -> List[dict]:
        def func():
            cn = self._get_collection()
            # args = {} if ids is None else {"_id": {"$in": [ObjectId(id_) for id_ in ids]}}
            args = {} if ids is None else {"_id": {"$in": ids}}
            cursor = cn.find(args, limit=limit)
            return [d for d in cursor]
        return self._query_wrapper(func)


def get_mongodb_records(database: str,
                        collection: str,
                        db_config: dict,
                        ids: Optional[Union[str, List[str]]] = None,
                        limit: int = 0) \
        -> Union[dict, List[dict], None]:
    """Get one or more MongoDB records from a single ID or list of IDs"""
    assert ids is None or isinstance(ids, (str, list))

    engine = MongoDBEngine(db_config, database=database, collection=collection)
    if ids is None:
        return engine.find_many(limit=limit)
    if isinstance(ids, str):
        return engine.find_one(ids)
    if isinstance(ids, list):
        return engine.find_many(ids=ids, limit=limit)
