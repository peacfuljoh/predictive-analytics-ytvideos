"""Raw data API"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from db_engines.mysql_engine import MySQLEngine
from db_engines.mongodb_engine import MongoDBEngine
from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_MONGO_CONFIG
from routes import (router_base, router_rawdata, router_prefeatures, router_vocabulary, router_config, router_features,
                    router_models, router_mongo)



# app init
app = FastAPI()
app.mysql_engine = None # set on startup
app.mongodb_engine = None # set on startup

# event handlers
@app.on_event("startup")
def startup_db_client():
    if 0:
        app.mongodb_engine = MongoDBEngine(DB_MONGO_CONFIG, verbose=True)
        app.mysql_engine = MySQLEngine(DB_MYSQL_CONFIG)

@app.on_event("shutdown")
def shutdown_db_client():
    del app.mysql_engine
    del app.mongodb_engine

# routes
app.include_router(router_base, tags=["base"])
app.include_router(router_rawdata, tags=["rawdata"], prefix="/rawdata")
app.include_router(router_prefeatures, tags=["prefeatures"], prefix='/prefeatures')
app.include_router(router_vocabulary, tags=["vocabulary"], prefix='/vocabulary')
app.include_router(router_config, tags=["config"], prefix='/config')
app.include_router(router_features, tags=["features"], prefix='/features')
app.include_router(router_models, tags=["models"], prefix='/model')
app.include_router(router_mongo, tags=["mongo"], prefix='/mongo')

# OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="YouTube Predictive Analytics API",
        version="0.0.1",
        summary="Summary",
        description="Description",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
