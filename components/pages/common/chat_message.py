import streamlit as st

from components.pages.common.session import append_session, get_session_or_init, set_session


def print_message(message: str, role: str):
    with st.chat_message(role):
        st.markdown(message)


def print_message_and_save(message_group_key: str, message: str, role: str):
    print_message(message, role)
    save_message(message_group_key, message, role)


def save_message(message_group_key: str, message: str, role: str):
    append_session(f'{message_group_key}_messages', {'message': message, 'role': role})


def print_message_history(message_group_key: str):
    for message in get_session_or_init(f'{message_group_key}_messages', []):
        print_message(message['message'], message['role'])


def clear_message_history(message_group_key: str):
    set_session(f'{message_group_key}_messages', [])
