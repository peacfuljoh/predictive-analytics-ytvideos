from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Book, BookUpdate

router = APIRouter()

@router.post("/", response_description="Create a new book", status_code=status.HTTP_201_CREATED, response_model=Book)
def create_book(request: Request, book: Book = Body(...)):
    """Insert a new book into the database"""
    book = jsonable_encoder(book)
    # print(book)
    if (book_ := request.app.database["books"].find_one({"_id": book['_id']})) is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Book with ID {book_['_id']} already exists.")

    new_book = request.app.database["books"].insert_one(book)
    created_book = request.app.database["books"].find_one({"_id": new_book.inserted_id})

    return created_book

@router.get("/", response_description="List all books", response_model=List[Book])
def list_books(request: Request):
    """Return the books in the database"""
    books = list(request.app.database["books"].find(limit=100))

    return books

@router.get("/{id}", response_description="Get a single book by id", response_model=Book)
def find_book(id: str, request: Request):
    """Find a book by its unique ID"""
    if (book := request.app.database["books"].find_one({"_id": id})) is not None:
        return book

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")

@router.put("/{id}", response_description="Update a book", response_model=Book)
def update_book(id: str, request: Request, book: BookUpdate = Body(...)):
    """Update the information for a book specified by its ID"""
    book = {k: v for k, v in book.model_dump().items() if v is not None}

    if len(book) >= 1:
        update_result = request.app.database["books"].update_one({"_id": id}, {"$set": book})
        if update_result.modified_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")

    if (existing_book := request.app.database["books"].find_one({"_id": id})) is not None:
        return existing_book

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")

@router.delete("/{id}", response_description="Delete a book")
def delete_book(id: str, request: Request, response: Response):
    """Delete the book with a specified ID from the database"""
    delete_result = request.app.database["books"].delete_one({"_id": id})

    if delete_result.deleted_count == 1:
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Book with ID {id} not found")