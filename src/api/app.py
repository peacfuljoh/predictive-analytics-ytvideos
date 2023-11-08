"""Raw data API"""

from fastapi import FastAPI
from routes import router_root, router_rawdata, router_prefeatures

from db_engines.mysql_engine import MySQLEngine
from db_engines.mongodb_engine import MongoDBEngine
from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_MONGO_CONFIG



# app init
app = FastAPI()
app.mysql_engine = None # set on startup
app.mongodb_engine = None # set on startup

# app event handlers
@app.on_event("startup")
def startup_db_client():
    app.mongodb_engine = MongoDBEngine(DB_MONGO_CONFIG)
    app.mysql_engine = MySQLEngine(DB_MYSQL_CONFIG)

@app.on_event("shutdown")
def shutdown_db_client():
    del app.mysql_engine
    del app.mongodb_engine

# add routes
app.include_router(router_root, tags=["root"])
app.include_router(router_rawdata, tags=["rawdata"], prefix="/rawdata")
app.include_router(router_prefeatures, tags=["prefeatures"], prefix='/prefeatures')
