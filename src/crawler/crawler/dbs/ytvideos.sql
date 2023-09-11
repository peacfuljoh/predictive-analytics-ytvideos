CREATE DATABASE IF NOT EXISTS ytvideos;
USE ytvideos;
CREATE TABLE IF NOT EXISTS users (
    username VARCHAR(50) PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS video_meta (
    video_id VARCHAR(20) PRIMARY KEY,
    username VARCHAR(50),
    title VARCHAR(100),
    upload_date TIMESTAMP,
    duration SMALLINT UNSIGNED,
    keywords VARCHAR(500),
    description VARCHAR(500),
    thumbnail_url VARCHAR(100),
    tags VARCHAR(500)
);
CREATE TABLE IF NOT EXISTS video_stats (
    video_id VARCHAR(20),
    date_accessed TIMESTAMP,
    like_count INT UNSIGNED,
    view_count INT UNSIGNED,
    subscriber_count INT UNSIGNED,
    comment_count SMALLINT UNSIGNED,
    comment VARCHAR(1000),
    FOREIGN KEY(video_id) REFERENCES video_meta(video_id),
    PRIMARY KEY(video_id, date_accessed)
);