# Predictive analytics for YouTube content creators

THIS PROJECT IS UNDER CONSTRUCTION. IT'S CURRENTLY HIGHLY MODULAR ALBEIT MONOLITHIC. ONCE READY, IT WILL BE PARTITIONED INTO DISTINCT SERVICES. AFTER THAT, IT WILL BE MIGRATED TO AN AWS SERVERLESS ARCHITECTURE. STAY TUNED!!

## Summary

This repo contains an end-to-end, full-stack web application implementing predictive analytics for YouTube (YT) videos. The goal is to predict future stats (e.g. likes, comments, views) from current stats, metadata (e.g. subscribers), text (e.g. title, description, tags, keywords), and the video's thumbnail image.

1. Data collection: A backend web crawler process regularly pulls video stats and other info for desired YT channels and stores them in MySQL tables.
2. Featurization: An ETL pipeline converts these records into "prefeatures" that are fed through a second ETL pipeline to produce "features", stored in MongoDB collections. Prefeaturization is sample-specific while featurization involves creating a dictionary for the text features from a desired subset of videos and users, and thus requires processing the entire dataset together.
3. Model training: A desired feature set is used to train a regression model that predicts future video stats from stats at a previous point in time.
4. Front-end dashboard: A Flask-served dashboard enables exploration of the database (slicing and filtering as desired), training models with desired feature subsets, and visualizing predictions.

The pipeline components use Python generators wherever possible to minimize memory use. This is a common data streaming technique when working with large-scale document datasets and is standard practice in text-processing libraries like Gensim (used in this project).

All steps in the pipeline have associated JSON-format configs that are stored alongside the corresponding ETL output artifacts. This enables feature and model versioning (a.k.a. provenance) and therefore reproducibility. If all data stores (not raw data!) are emptied, the pipeline can be re-run with the same config options to produce the same outputs.

Raw data is continuously crawled for the latest videos, so we can run predictive analytics in real-time!


### Frameworks

Frameworks/technologies used (initial dev, on-prem):
- Database: MySQL, MongoDB, Redis
- Web: Scrapy, Flask (w/ Jinja, Plotly)
- API: FastAPI (w/ OpenAPI spec)
- Text: Gensim
- Automation/Deployment: Make, Github Actions
- Packaging: Poetry

Frameworks/technologies used (cloud, serverless):
- Database: RDS, DynamoDB
- Web: Scrapy, Flask
- API: FastAPI + API Gateway
- ETL pipelines: Lambda
- Automation/Deployment: CodeCommit, CodeBuild, CodeDeploy, CodePipeline


## Steps to set up the repo

These steps were implemented in Ubuntu 20.04.1 on an Intel NUC.

### General

1. Clone this repo: `git clone <HTTPS url>`
2. Setup conda env from requirements file: `conda env create -f environment.yml`
3. Install MySQL Server for Linux (see https://dev.mysql.com/doc/mysql-apt-repo-quick-guide/en/#apt-repo-fresh-install)

### Database 

5. Create config file somewhere outside of the repo and set its full path via `GLOBAL_CONFIG_PATH` in `src/crawler/crawler/config.py`.
6. Set up the database and inject username info by running `scripts/setup_videos_db.py`. These are the names of users whose videos will be crawled.

### Run the crawler

There are two spiders, one for reading the most recent video_id information from users' "videos" pages 
(`"yt-latest-video-ids"`) and another for reading stats off of the most recent videos themselves (`"yt-video-stats"`).
These are called once every so often to update the database with video metadata and stats. This is done with a polite
delay between crawls (e.g. 10 seconds). This avoids spamming the target domain's servers and getting IP banned.

7. Run the timed crawler script: `src/crawler/main.py`.


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
    "DB_VIDEOS_TABLES": {
      "users": "users",
      "meta": "video_meta",
      "stats": "video_stats"
    }
  },
  "AUTOCRAWL_CONFIG": {
    "REPEAT_INTVL": 3600
  }
}
```


## Helpful links

Managing Conda environments: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

MySQL setup and Python API tutorial: https://realpython.com/python-mysql/

MongoDB setup: https://www.mongodb.com/languages/python/pymongo-tutorial

MongoDB on Ubuntu: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/#std-label-install-mdb-community-ubuntu

Randomized delays between scrapy crawls: https://scrapeops.io/python-scrapy-playbook/scrapy-delay-between-requests/


## Make CLI commands

There are several `make` commands that are utilities for running processes from the command line.
- `make crawl`
- `make dashboard`
- `make test`

These activate the conda env, cd into a directory, and then run a script. See the `Makefile` for more info.


## Note on web crawling

YouTube has a dedicated data API that should be a first point of contact with their content statistics. 
In contrast, this repo runs scrapy spiders to read recently-posted videos' stats for a small number of desired users 
at a slow pace. Doing this slowly is polite as it doesn't overwhelm the target domain's servers. This also helps 
to avoid an IP ban. Please use this functionality responsibly.



## Machine learning

### Regression model and features

The predictive analytics model performs regression between a video's statistics at two points in time. The input is 
an N-dimensional vector and the output is a small vector of statistics to predict.

Inputs:
- `bow`: sparse bag-of-words representation of the text content of the video, e.g. derived from title, description, tags, and comments
- `subscriber count`: generally a large number that doesn't change much over time, mostly used to account for stats scale
- `username`: a one-hot-coded vector to distinguish between users when training a global (multi-user) model
- `time_after_upload_src`: seconds since video was first published at a source time
- `stats at source time`: numerical values for comments, likes, and views athe source time
- `time_after_upload_tgt`: same as for source time, but at a later target time

Outputs:
- `stats at target time`

### Basic model: L2-regularized linear regression with random linear projection (LR-RLP)

In the basic regression model, a data-independent, random (sparse) linear projection is used to reduce the 
dimensionality of the bag-of-words vectors from, e.g. 40,000 to 512. Given how sparse the bow vectors are in the first
place, this is very likely quasi-lossless, especially since we are not using word position information to represent
the text.

Feature vectors for training are created by sampling pairs of measurement times (e.g. 50-100 per video) to create 
the input list described above. These features are stacked into a single vector per sample. The resulting dataset
is fed to a linear regression model with L2 regularization (e.g. `err = err_LS + alpha * || w ||^2`) where the 
regularization parameter `alpha` is tuned with cross-validation.

Given that this is a linear model, the predicted stats will be linear in time. The intercept of these lines shifts 
to account for the stat value at the source time, but their slopes do not. This is because the slope is determined
solely by the coefficient corresponding to the `time_after_upload_tgt` feature.

Better accuracy is observed with user-specific models rather than trying to learn a global model across all users.

Under experimentation: One can create a non-linear extrapolator (analogous to a numerical solver) with this model 
where evaluations of the
model over successive, small time increments are able to trace out a non-linear prediction of stats.
This does a surprisingly good job of tracking some stats curves in some cases, but needs more work.

### Advanced model: generalized additive model (GAM) \[not implemented\]

A more advanced model could attempt to learn a temporal profile of stats values for each word or embedded-word feature.
These profiles could then be combined according to the other features to make non-linear predictions into the future.

### Making predictions

This model takes both static metadata (`bow`, `subscriber count`, `username`) and dynamic data (`time_after_upload_src`, `stats`) into 
account to make predictions. During inference all inputs except for 
`time_after_upload_tgt` are held fixed while `time_after_upload_tgt` is swept over future times to predict stats 
arbitrarily far into the future. Of course, the accuracy will decrease as the gap between source and target times 
increases.

### Data formats

A dataset consists of a list of users and raw data from their videos. A vocabulary is learned from the text content of 
all videos, which is then used to construct the `bow` feature representation for each video.

In practice, the `bow` component of the input is first embedded into a 1000-dimensional dense vector using 
linear dimensionality reduction or other techniques (e.g. fastText, word2vec). This simplifies and speeds up training
significantly.


## APIs

APIs are implemented as FastAPI processes with an OpenAPI spec and associated Swagger docs.

To launch the `raw_data` API process on the command line (at repo root level): 
`PYTHONPATH=/home/nuc/crawler python src/api/raw_data/main.py`


### Raw data
