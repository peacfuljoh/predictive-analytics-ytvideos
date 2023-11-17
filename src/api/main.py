"""
Dev entry point for API.

To run at command line from repo root:
    PYTHONPATH=$(pwd) python src/api/main.py
"""

import time

import uvicorn

from app import app  # to register refactoring changes

from src.crawler.crawler.config import API_CONFIG


if __name__ == "__main__":
    while 1:
        uvicorn.run("app:app", host=API_CONFIG['host'], port=API_CONFIG['port'], reload=False)
        # try:
        #     uvicorn.run("app:app", host=API_CONFIG['host'], port=API_CONFIG['port'], reload=False)
        # except:
        #     print('\n\nAPI crashed. Restarting.\n\n')
        time.sleep(1)
