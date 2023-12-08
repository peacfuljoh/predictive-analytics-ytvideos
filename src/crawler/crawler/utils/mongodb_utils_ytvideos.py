"""Utils for interfacing with the MongoDB database"""

from ytpa_utils.misc_utils import fetch_data_at_url
from db_engines.mongodb_engine import MongoDBEngine


def save_image_to_db(database: str,
                     collection: str,
                     db_config: dict,
                     video_id: str,
                     image_data: bytes,
                     verbose: bool = False):
    """Insert image to MongoDB collection"""
    engine = MongoDBEngine(db_config, database=database, collection=collection, verbose=verbose)
    record = {'_id': video_id, 'img': image_data}
    engine.insert_one(record)

def fetch_url_and_save_image(database: str,
                             collection: str,
                             db_config: dict,
                             video_id: str,
                             url: str,
                             delay: int = 0,
                             verbose: bool = False):
    """Fetch image and insert into MongoDB collection, but only if it isn't already there."""
    engine = MongoDBEngine(db_config, database=database, collection=collection, verbose=verbose)
    if engine.find_one_by_id(id=video_id) is None:
        record = {'_id': video_id, 'img': fetch_data_at_url(url, delay=delay)}
        engine.insert_one(record)

