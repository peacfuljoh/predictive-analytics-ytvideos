"""Raw data API"""

from fastapi import FastAPI
from routes import router_root, router_rawdata

from db_engines.mysql_engine import MySQLEngine
from src.crawler.crawler.config import DB_MYSQL_CONFIG



# app init
app = FastAPI()
app.mysql_engine = None # set on startup

# app event handlers
@app.on_event("startup")
def startup_db_client():
    # app.mongodb_engine = MongoDBEngine(DB_MONGO_CONFIG)
    app.mysql_engine = MySQLEngine(DB_MYSQL_CONFIG)

@app.on_event("shutdown")
def shutdown_db_client():
    del app.mysql_engine

# add routes
app.include_router(router_root, tags=["root"])
app.include_router(router_rawdata, tags=["rawdata"], prefix="/rawdata")
