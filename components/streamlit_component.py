from typing import Any

import streamlit as st


def init_session_singleton(session_key: str, init_value: Any):
    if session_key not in st.session_state:
        st.session_state[session_key] = init_value


def save_message(message: str, role: str):
    st.session_state['messages'].append({'message': message, 'role': role})


def send_message(message: str, role: str, save: bool = True):
    with st.chat_message(role):
        st.markdown(message)

    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state['messages']:
        send_message(message['message'], message['role'], save=False)

