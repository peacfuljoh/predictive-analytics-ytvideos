# crawler

Web crawling for YouTube videos.


## Steps to set up the crawler

### General

1. Clone this repo: `git clone <HTTPS url>`
2. Setup conda env from requirements file: `conda env create -f environment.yml`
3. Install MySQL Server for Linux (see https://dev.mysql.com/doc/mysql-apt-repo-quick-guide/en/#apt-repo-fresh-install)

### Database 

5. Create config file somewhere outside of the repo and set its full path via `GLOBAL_CONFIG_PATH` in `src/crawler/crawler/config.py`.
6. Set up the database by running `scripts/setup_videos_db.py`.


## Config file

The file `config.json` is kept outside of the repo and contains sensitive db info (credentials) and paths, 
plus other non-sensitive info needed by the crawler.

```json
{
  "DB_CONFIG": {
    "host": "localhost",
    "user": "<USER>",
    "password": "<PASSWORD>"
  },
  "DATA_ROOT": "<DATA_DIR_PATH>",
  "REPO_ROOT": "<REPO_DIR_PATH>",
  "DB_INFO": {
    "DB_VIDEOS_DATABASE": "ytvideos",
    "DB_VIDEOS_TABLENAMES": {
      "users": "users",
      "meta": "video_meta",
      "stats": "video_stats"
    }
  }
}
```


## Helpful links

Managing Conda environments: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

MySQL setup and Python API tutorial: https://realpython.com/python-mysql/

Randomized delays between scrapy crawls: https://scrapeops.io/python-scrapy-playbook/scrapy-delay-between-requests/

