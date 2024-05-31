import streamlit as st

from components.pages.common.session import set_session
from components.util.regex_util import chk_url_isvalid


def on_button_click(url: str):
    chk_result = False
    if not url:
        st.error("SiteMap URL is Empty")
    elif chk_url_isvalid(url) is False:
        with st.sidebar:
            st.error("Invalid URL")
    elif ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a SiteMap URL")
    else:
        chk_result = True

    set_session('run_search_sitemap', chk_result)
