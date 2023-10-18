"""Raw data API"""

from fastapi import FastAPI
from routes import router

from src.crawler.crawler.utils.mysql_engine import MySQLEngine
from src.crawler.crawler.config import DB_CONFIG


app = FastAPI()
app.mysql_engine = None # set on startup

@app.on_event("startup")
def startup_db_client():
    # app.mongodb_engine = MongoDBEngine(DB_MONGO_CONFIG)
    app.mysql_engine = MySQLEngine(DB_CONFIG)

@app.on_event("shutdown")
def shutdown_db_client():
    del app.mysql_engine

app.include_router(router, tags=["raw_data"], prefix="/raw_data")
