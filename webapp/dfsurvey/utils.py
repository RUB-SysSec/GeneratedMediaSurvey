"""Various utility scripts."""
from typing import Tuple


def remove_double_whitespace(text: str) -> str:
    """ Remove double whitespace from text.

    Args:
        test (str): Text to clean.

    Returns:
        The cleaned text.
    """
    cleaned = ""
    whitespace = False
    for c in text:
        if c == " " and whitespace:
            continue
        elif c == " " and not whitespace:
            whitespace = True
        else:
            whitespace = False

        cleaned += c

    return cleaned


def make_about_equal(a: str, b: str) -> Tuple[str, str]:
    """Given two text make them about equal length while retaining sentences.
    Args:
        a (str): Text a.
        b (str): Text b.
    Returns:
        Both text.
    """
    if len(a) < len(b):
        b = b[:b.find(".", len(a)) + 1]
    else:
        a = a[:a.find(".", len(b)) + 1]

    return a, b


def strip_including(string: str, substring: str) -> str:
    return string[string.rfind(substring) + 1:]
