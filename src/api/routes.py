"""Routes for video raw_data store"""

from typing import List, Tuple
from pprint import pprint

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder

# from models import Book, BookUpdate

from src.crawler.crawler.utils.mysql_engine import MySQLEngine
from src.crawler.crawler.utils.misc_utils import make_sql_query
from src.crawler.crawler.config import DB_INFO


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']


router_root = APIRouter()
router_rawdata = APIRouter()



""" Setup """
def get_mysql_engine_and_tablename(request: Request,
                                   key: str) \
        -> Tuple[MySQLEngine, str]:
    """Get MySQL engine and tablename"""
    engine = request.app.mysql_engine
    tablename = DB_VIDEOS_TABLES[key]

    return engine, tablename



""" Root """
@router_root.get("/", response_description="Test for liveness", response_model=str)
def get_usernames(request: Request):
    """Test for liveness"""
    return "The app is running"



""" Raw data """
@router_rawdata.get("/users", response_description="Get video usernames", response_model=List[str])
def get_video_usernames(request: Request):
    """Get all usernames"""
    # db access
    engine, tablename = get_mysql_engine_and_tablename(request, 'users')

    # query
    query = make_sql_query(tablename)

    # get records
    ids: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)
    ids: List[str] = [id_[0] for id_ in ids]

    return ids

@router_rawdata.post("/meta", response_description="Get video metadata", response_model=List[tuple])
def get_video_metadata(request: Request, opts: dict):
    """Get video meta information"""
    pprint(opts)

    # db access
    engine, tablename = get_mysql_engine_and_tablename(request, 'meta')

    # query
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))

    # get records
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)

    return records

@router_rawdata.post("/stats", response_description="Get video stats", response_model=List[tuple])
def get_video_stats(request: Request, opts: dict):
    """Get video meta information"""
    pprint(opts)

    # db access
    engine, tablename = get_mysql_engine_and_tablename(request, 'stats')

    # query
    query = make_sql_query(tablename, opts.get('cols'), opts.get('where'), opts.get('limit'))

    # get records
    records: List[tuple] = engine.select_records(DB_VIDEOS_DATABASE, query)

    return records







# @router.post("/", response_description="Create a new book", status_code=status.HTTP_201_CREATED, response_model=Book)
# def create_book(request: Request, book: Book = Body(...)):
#     """Insert a new book into the database"""
#     book = jsonable_encoder(book)
#     # print(book)
#     if (book_ := request.app.database["books"].find_one_by_id({"_id": book['_id']})) is not None:
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Book with ID {book_['_id']} already exists.")
#
#     new_book = request.app.database["books"].insert_one(book)
#     created_book = request.app.database["books"].find_one_by_id({"_id": new_book.inserted_id})
#
#     return created_book
#
# @router.get("/", response_description="List all books", response_model=List[Book])
# def list_books(request: Request):
#     """Return the books in the database"""
#     books = list(request.app.database["books"].find(limit=100))
#
#     return books
#
# @router.get("/{id}", response_description="Get a single book by id", response_model=Book)
# def find_book(id: str, request: Request):
#     """Find a book by its unique ID"""
#     if (book := request.app.database["books"].find_one_by_id({"_id": id})) is not None:
#         return book
#
#     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")
#
# @router.put("/{id}", response_description="Update a book", response_model=Book)
# def update_book(id: str, request: Request, book: BookUpdate = Body(...)):
#     """Update the information for a book specified by its ID"""
#     book = {k: v for k, v in book.model_dump().items() if v is not None}
#
#     if len(book) >= 1:
#         update_result = request.app.database["books"].update_one({"_id": id}, {"$set": book})
#         if update_result.modified_count == 0:
#             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")
#
#     if (existing_book := request.app.database["books"].find_one_by_id({"_id": id})) is not None:
#         return existing_book
#
#     raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")
