import re

__url_pattern = ("^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9("
                 ")@:%_\\+.~#?&\\/=]*)$")


def chk_url_isvalid(url: str) -> bool:
    return re.match(__url_pattern, url) is not None
