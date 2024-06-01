from typing import Any

import streamlit as st


def set_session(session_key: str, session_value: Any):
    st.session_state[session_key] = session_value


def get_session(session_key: str) -> Any:
    return st.session_state[session_key]


def get_session_or_init(session_key: str, init_value: Any) -> Any:
    if session_key not in st.session_state:
        st.session_state[session_key] = init_value

    return st.session_state[session_key]


def clear_session(session_key: str):
    del st.session_state[session_key]


def append_session(session_key: str, session_value: Any):
    st.session_state[session_key].append(session_value)
