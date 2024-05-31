from typing import Any

import streamlit as st


def set_session(session_key: str, session_value: Any):
    st.session_state[session_key] = session_value
