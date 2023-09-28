

from pprint import pprint

from pymongo.collection import ObjectId

from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine


DB_VIDEOS_NOSQL_DATABASE = DB_INFO['DB_VIDEOS_NOSQL_DATABASE'] # NoSQL thumbnails
DB_VIDEOS_NOSQL_COLLECTIONS = DB_INFO['DB_VIDEOS_NOSQL_COLLECTIONS']
DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


def inspect_mongodb(inject_data: bool = False):
    print(DB_FEATURES_NOSQL_DATABASE)
    print(DB_FEATURES_NOSQL_COLLECTIONS)

    engine = MongoDBEngine(DB_MONGO_CONFIG)

    dbs = engine.get_all_collections(DB_FEATURES_NOSQL_DATABASE)
    pprint(dbs)

    engine.set_db_info(DB_FEATURES_NOSQL_DATABASE, DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'])

    ids = engine.get_ids()
    print(len(ids))
    # print(ids)

    data = engine.find_many()
    print(len(data))
    pprint(data[:10])


def delete_records_by_id():
    if input('About to delete records. Are you sure (yes/no)? ') == 'yes':
        ids = ['$set']
        engine = MongoDBEngine(DB_MONGO_CONFIG,
                               database=DB_FEATURES_NOSQL_DATABASE,
                               collection=DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'])
        engine.delete_many(ids=ids)
        # engine.delete_all_records('yes')



if __name__ == '__main__':
    # delete_records_by_id()
    inspect_mongodb(inject_data=True)
