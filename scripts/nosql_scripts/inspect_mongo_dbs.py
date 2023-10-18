"""Inspect mongodb database."""

from pprint import pprint

from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine


DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE'] # NoSQL thumbnails
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']
DB_MODELS_NOSQL_DATABASE = DB_INFO['DB_MODELS_NOSQL_DATABASE'] # NoSQL models
DB_MODELS_NOSQL_COLLECTIONS = DB_INFO['DB_MODELS_NOSQL_COLLECTIONS']



def inspect_mongodb(inject_data: bool = False):
    engine = MongoDBEngine(DB_MONGO_CONFIG)

    cns = engine.get_all_collections()
    pprint(cns)

    dbs = [DB_VIDEOS_NOSQL_DATABASE, DB_FEATURES_NOSQL_DATABASE, DB_MODELS_NOSQL_DATABASE]
    cns = [DB_VIDEOS_NOSQL_COLLECTIONS, DB_FEATURES_NOSQL_COLLECTIONS, DB_MODELS_NOSQL_COLLECTIONS]

    print('')
    pprint(dbs)
    pprint(cns)
    print('')

    for database, collections in zip(dbs, cns):
        for collection in collections.values():
            engine.set_db_info(database=database, collection=collection)

            print('Database: ' + database)
            print('Collection: ' + collection)

            ids = engine.get_ids()
            n_ids = len(ids)
            print(f'n_ids: {n_ids}')
            print(f'ids[:5]: {ids[:5]}...')

            if n_ids > 0:
                id_ = ids[0]
                data = engine.find_one_by_id(id_)
                print(f'Record has keys: {list(data.keys())}')

            print('')


def delete_records_by_id():
    if input('About to delete records. Are you sure (yes/no)? ') == 'yes':
        ids = ['$set']
        engine = MongoDBEngine(DB_MONGO_CONFIG,
                               database=DB_FEATURES_NOSQL_DATABASE,
                               collection=DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'])
        engine.delete_many(ids=ids)
        # engine.delete_all_records('yes')

def delete_all_records():
    if input('About to delete records. Are you sure (yes/no)? ') == 'yes':
        for _, collection in DB_FEATURES_NOSQL_COLLECTIONS.items():
            engine = MongoDBEngine(DB_MONGO_CONFIG,
                                   database=DB_FEATURES_NOSQL_DATABASE,
                                   collection=collection)
            engine.delete_all_records('yes')



if __name__ == '__main__':
    inspect_mongodb(inject_data=False)
    # delete_records_by_id()
    # delete_all_records()
