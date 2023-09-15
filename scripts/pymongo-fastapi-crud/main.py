from fastapi import FastAPI
from dotenv import dotenv_values
from pymongo import MongoClient
from routes import router as book_router

config = dotenv_values(".env")

app = FastAPI()
app.mongodb_client = None # set on startup
app.database = None # set on startup

@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient(config['MONGODB_HOST'], int(config['MONGODB_PORT']))
    app.database = app.mongodb_client[config["DB_NAME"]]
    print(app.mongodb_client)
    print(app.database)
    print("Connected to the MongoDB database!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()

app.include_router(book_router, tags=["books"], prefix="/book")
