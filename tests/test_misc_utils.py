
from src.crawler.crawler.utils.misc_utils import flatten_dict_of_lists, make_videos_page_urls_from_usernames, \
    convert_num_str_to_int
from utils_for_tests import assert_lists_match


def test_flatten_dict_of_lists():
    in_ = {
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    }
    exp = [1, 2, 3, 'x', 'y', 'z']

    res = flatten_dict_of_lists(in_)

    assert_lists_match(exp, res)

def test_make_videos_page_urls_from_username():
    in_ = ['bb', 'cc', '583']
    exp = [
        "https://www.youtube.com/@bb/videos",
        "https://www.youtube.com/@cc/videos",
        "https://www.youtube.com/@583/videos"
    ]

    res = make_videos_page_urls_from_usernames(in_)

    assert_lists_match(exp, res)

def test_convert_num_str_to_int():
    in_ = ['345', '2525200', '34,659', '483,203,991', '50K', '33.2K', '4.3M', '50M']
    exp = [345, 2525200, 34659, 483203991, 50000, 33200, 4300000, 50000000]
    for a, b in zip(in_, exp):
        res = convert_num_str_to_int(a)
        assert res == b


if __name__ == '__main__':
    test_flatten_dict_of_lists()
    test_make_videos_page_urls_from_username()
    test_convert_num_str_to_int()