# Predictive analytics for YouTube content creators


## Summary

### Goal

The goal with this repository is to create an end-to-end, full-stack web application implementing predictive 
analytics for YouTube (YT) videos. 
The system predicts future stats (e.g. likes, comments, views) from current stats, metadata (e.g. subscribers), 
text (e.g. title, description, tags, keywords), and the video's thumbnail image.

Note: as of 2/10/2024, the implementation is local, modular, and implements a seq2seq deep learning model.
It does not yet have an MLOps serving infrastructure and migration to AWS is TBD (waiting on further model dev).


### Software components

1. Data collection: A backend web crawler process regularly pulls video stats and other info for 
desired YT channels and stores them in MySQL tables.
2. Featurization: An ETL pipeline converts these records into "prefeatures" that are fed through a second ETL 
pipeline to produce "features", stored in MongoDB collections. 
Prefeaturization is sample-specific while featurization involves creating a dictionary for the text features
from a desired subset of videos and users, and thus requires processing the entire dataset together.
3. Model training: A desired feature set is used to train a regression model that predicts future video stats from 
past and current stats.
4. Front-end dashboard: A Flask-served dashboard enables exploration of the database (slicing and filtering as 
desired), training models with desired feature subsets, and visualizing predictions. (Note: not started yet)

### Other details

The pipeline components use Python generators wherever possible to limit memory use. 
This is a common data streaming technique when working with large-scale document datasets and is standard practice 
in text-processing libraries like Gensim (used in this project).

As in MLOps best practices, all steps in the pipeline have associated JSON-format configs that are stored alongside 
the corresponding ETL output artifacts. 
This enables feature and model versioning (a.k.a. provenance) and therefore reproducibility. 
If all derived data stores (not raw data!) are emptied, the pipeline can be re-run with the same config options to 
produce the same outputs.

Raw data is continuously crawled for the latest videos, so we can run predictive analytics in real-time!

Frameworks/technologies used (initial dev, on-prem):
- Database: MySQL, MongoDB(, Redis)
- Web: Scrapy, (Flask (w/ Jinja, Plotly))
- API: FastAPI
- Text: Gensim
- Automation/Deployment: Make, Github Actions
- Packaging: Poetry

On AWS, the various pipelines will be implemented as Lambda services. 







## Steps to set up the repo

Initial development was in Ubuntu 20.04.1 on an Intel NUC.

### General

1. Clone this repo: `git clone <HTTPS url>`
2. Setup conda env from requirements file: `conda env create -f environment.yml`
3. Install MySQL Server for Linux (see https://dev.mysql.com/doc/mysql-apt-repo-quick-guide/en/#apt-repo-fresh-install)

### Database 

5. Create config file somewhere outside of the repo and set its full path via `GLOBAL_CONFIG_PATH` in `src/crawler/crawler/config.py`.
6. Set up the database and inject username info by running `scripts/setup_videos_db.py`.
These are the names of users whose videos will be crawled.

### Run the crawler

There are two spiders, one for reading the most recent video_id information from users' "videos" pages 
(`"yt-latest-video-ids"`) and another for reading stats off of the most recent videos themselves (`"yt-video-stats"`).
These are called once every so often to update the database with video metadata and stats. This is done with a polite
delay between crawls (e.g. 10 seconds). This avoids spamming the target domain's servers and getting IP banned.

7. Run the timed crawler script: `src/crawler/main.py`.

#### Note on web crawling

YouTube has a dedicated data API that should be a first point of contact with their content statistics. 
In contrast, this repo runs scrapy spiders to read recently-posted videos' stats for a small number of desired users 
at a slow pace. Doing this slowly is polite as it doesn't overwhelm the target domain's servers. This also helps 
to avoid an IP ban. Please use this functionality responsibly.

### Packaging

This project spans several Python packages (maintained via Poetry and available on PyPI):

- `ytpa_utils`: general utils used across the entire project
- `ytpa_api_utils`: API-specific tools (e.g. for handling communication over websockets)
- `db_engines`: tools for working with MySQL and MongoDB databases
- `jtorch_utils`: helpful tools for simplifying model training and experimentation

### Config file

The file `config.json` is kept outside of the repo and contains sensitive db info (credentials) and paths, 
plus other non-sensitive info needed by the crawler and ETL pipelines.

```json
{
  "DB_MYSQL_CONFIG": {
    "host": "localhost",
    "user": "root",
    "password": "<MYSQL_PASSWORD>"
  },
  "DB_MONGO_CONFIG": {
  	"host": "localhost",
  	"port": 27017
  },
  "API_CONFIG": {
    "host": "<IP_ADDRESS>",
    "port": "<CUSTOM_PORT>"
  },
  "DATA_ROOT": "<DATA_DIR_PATH>",
  "REPO_ROOT": "<REPO_DIR_PATH>",
  "DB_INFO": {
    "DB_VIDEOS_DATABASE": "ytvideos",
    "DB_VIDEOS_TABLES": {
      "users": "users",
      "meta": "video_meta",
      "stats": "video_stats"
    },
    "DB_VIDEOS_NOSQL_DATABASE": "ytvideos",
    "DB_VIDEOS_NOSQL_COLLECTIONS": {
      "thumbnails": "thumbnails"
    },
    "DB_FEATURES_NOSQL_DATABASE": "ytfeatures",
    "DB_FEATURES_NOSQL_COLLECTIONS": {
      "prefeatures": "prefeatures",
      "features": "features",
      "vocabulary": "vocabulary",
      "etl_config_prefeatures": "etl_config_prefeatures",
      "etl_config_features": "etl_config_features",
      "etl_config_vocabulary": "etl_config_vocabulary"
    },
    "DB_MODELS_NOSQL_DATABASE": "ytmodels",
    "DB_MODELS_NOSQL_COLLECTIONS": {
      "models": "models",
      "meta": "metadata"
    }
  },
  "AUTOCRAWL_CONFIG": {
    "REPEAT_INTVL": 3600
  }
}
```




## Machine learning

A predictive analytics model performs regression between a video's statistics at two points in time. At each point
in time, we have an N-dimensional vector as input a small vector of statistics to predict as output.
Beyond this specification, the implementation of the model can take any form (e.g. linear, recurrent).

### Baseline model

A common-sense baseline model applies L2-regularized linear regression to random linearly projected feature vectors.
The input and output features are as follows.

Inputs:
- `bow`: sparse bag-of-words representation of the text content of the video, e.g. derived from title, description, tags, and comments
- `subscriber count`: generally a large number that doesn't change much over time, mostly used to account for stats scale
- `time_after_upload_src`: seconds since video was first published at a source time
- `stats at source time`: numerical values for comments, likes, and views at the source time
- `time_after_upload_tgt`: same as for source time, but at a later target time

Outputs:
- `stats at target time`

In the basic regression model, a data-independent, random (sparse) linear projection is used to reduce the 
dimensionality of the bag-of-words vectors from, e.g. 40,000 to 512. Given how sparse the bow vectors are in the first
place, this is very likely quasi-lossless, especially since we are not using word position information to represent
the text.

Feature vectors for training are created by sampling pairs of measurement times (e.g. 50-100 per video) to create 
the input list described above. These features are stacked into a single vector per sample. The resulting dataset
is fed to a linear regression model with L2 regularization (e.g. `err = err_LS + alpha * || w ||^2`) where the 
regularization parameter `alpha` is tuned with cross-validation.

#### Observations

This model takes both static metadata (`bow`, `subscriber count`, `username`) and dynamic data 
(`time_after_upload_src`, `stats`) into account to make predictions. During inference all inputs except for 
`time_after_upload_tgt` are held fixed while `time_after_upload_tgt` is swept over future times to predict stats 
arbitrarily far into the future. Of course, the accuracy will decrease as the gap between source and target times 
increases.

Given that this is a linear model, the predicted stats will be linear in time. The intercept of these lines shifts 
to account for the stat value at the source time, but their slopes do not. This is because the slope is determined
solely by the coefficient corresponding to the `time_after_upload_tgt` feature.

Better accuracy is observed with user-specific models rather than trying to learn a global model across all users.




### Recurrent model

In practice, we see a variety of patterns in the stats over time. Some have a slow, somewhat linear "rise time" while
others spike upwards and quickly plateau off. Given the diversity of these patterns and that it's not immediately clear
what determines them, a recurrent neural network (e.g. deep LSTM) is worth trying.

The LSTM model we used combines both the non-time-varying metadata (embedded into one vector per video) and the 
time-varying stats. Data is preprocessed to fill gaps corresponding to when the crawler 
was down and to resample to an exact 1-hour sampling interval. The subscriber count is included as non-time-varying
data.

The model consists of:
- per-feature normalizers: provides data as zero mean, unit standard deviation transformed input features
- embedding transformation: two linear layers with ReLU activations to project static features to a 10-dimensional vector
- one-step predictive LSTM: stats and compressed embeddings are concatenated and fed to the input of an LSTM that 
has stacked residual blocks, each one consisting of a 2-layer LSTM and skip connection. The last block doesn't have
the skip connection to encourage the model to learn more accurate long-term predictions. There are also linear layers
before and after the LSTM stack to map from the stats dimensionality to the LSTM's hidden size and back.

The target is the next time step's input (i.e. one-step-advanced normalized stats). 
Similar-length sequences are grouped and truncated to have the same length within their respective groups.
This way, during training, batches of same-length sequences can be sampled from the dataset.

At predict time, predictions are generated by priming the model with existing measurements and then autoregressively
generating new values over a time horizon of interest.




## API

The (internal) API is implemented with FastAPI. This accomplishes several things:
- Modularity/security: The API hides all database transactions from other microservices, i.e. the only way to 
interact with the MySQL and MongoDB stores is through API calls (REST or websocket). 
This means that only the API server needs to have db credentials. This isolates and secures the data.
- All db transactions are implemented in one go-to interface (raw data, features, models, etc.).
- New db-transaction endpoints are easy to develop and add.

You can browse REST endpoints on the server via the Swagger docs at `http://<HOST>:<PORT>/docs`. This will not, 
however, show spec info for websocket endpoints.

For more on FastAPI documentation, see https://www.linode.com/docs/guides/documenting-a-fastapi-app-with-openapi/.

### Access endpoints from the command line with curl

Example of a CURL request to get selected columns from filtered video `stats` records (and print nicely to command line
with `jq`):

```
curl \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"cols": ["video_id", "like_count", "view_count", "timestamp_accessed"], "where": {"timestamp_accessed": [["2023-10-19 05:00:00", "2023-10-19 05:05:00"]]}}' \
    http://<HOST>:<PORT>/rawdata/stats/pull \
    | jq
```





## Miscellaneous

### Helpful links

Managing Conda environments: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

MySQL setup and Python API tutorial: https://realpython.com/python-mysql/

Setting up MongoDB on Ubuntu: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/

Working with FastAPI and the MongoDB Python API (PyMongo): https://www.mongodb.com/languages/python/pymongo-tutorial

Randomized delays between scrapy crawls: https://scrapeops.io/python-scrapy-playbook/scrapy-delay-between-requests/


### Database engines

Once MongoDB is installed, you can run the interactive shell with `mongosh` on the command line. The linux process for
the database server is `mongod` and can be inspected with `sudo systemctl status mongod`.


### Make CLI commands

There are several `make` commands for running processes from the command line.
- `make crawl`
- `make dashboard`
- `make test`
- `make api`

These activate the conda env, cd into a directory, and then run a script. See the `Makefile` for more info.

