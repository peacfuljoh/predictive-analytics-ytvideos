


def assert_lists_match(exp: list, res: list):
    """Assert that list returned by a function call matches the expected list."""
    assert isinstance(res, list)
    assert len(exp) == len(res)
    assert all([exp[i] == res[i] for i in range(len(exp))])
