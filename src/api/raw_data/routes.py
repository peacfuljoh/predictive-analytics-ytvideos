"""Routes for video raw_data store"""

from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

# from models import Book, BookUpdate

from src.crawler.crawler.config import DB_INFO


DB_VIDEOS_DATABASE = DB_INFO['DB_VIDEOS_DATABASE']
DB_VIDEOS_TABLES = DB_INFO['DB_VIDEOS_TABLES']


router = APIRouter()



@router.get("/usernames/all", response_description="Get all usernames", response_model=List[str])
def get_usernames(request: Request):
    """Get all usernames"""
    engine = request.app.mysql_engine
    tablename = DB_VIDEOS_TABLES["users"]
    query = f'SELECT * FROM {tablename}'
    ids = engine.select_records(DB_VIDEOS_DATABASE, query)
    return ids









#
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
