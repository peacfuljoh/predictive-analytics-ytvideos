
from src.crawler.crawler.paths import USERNAMES_JSON_PATH
from src.crawler.crawler.utils.misc_utils import get_video_urls, get_video_urls_for_user


# multiple users
categories = None
usernames_desired = None

video_urls = get_video_urls(USERNAMES_JSON_PATH, categories=categories, usernames_desired=usernames_desired)
print(video_urls)

print('\n' + '*' * 20 + '\n')


# single user
username = 'twosetviolin'

video_ids, video_urls = get_video_urls_for_user(username)
for id, url in zip(video_ids, video_urls):
    print([id, url])
