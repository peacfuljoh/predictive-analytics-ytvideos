CREATE DATABASE IF NOT EXISTS ytvideos;
USE ytvideos;
CREATE TABLE IF NOT EXISTS users (
    username VARCHAR(50) PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS video_meta (
    video_id VARCHAR(20) PRIMARY KEY,
    username VARCHAR(50),
    title VARCHAR(200),
    upload_date DATE,
    duration SMALLINT UNSIGNED,
    keywords VARCHAR(1000),
    description VARCHAR(500),
    thumbnail_url VARCHAR(500),
    tags VARCHAR(500)
);
CREATE TABLE IF NOT EXISTS video_stats (
    video_id VARCHAR(20),
    timestamp_accessed TIMESTAMP(3),
    like_count INT UNSIGNED,
    view_count INT UNSIGNED,
    subscriber_count INT UNSIGNED,
    comment_count INT UNSIGNED,
    comment VARCHAR(1000),
    FOREIGN KEY(video_id) REFERENCES video_meta(video_id),
    PRIMARY KEY(video_id, timestamp_accessed)
);