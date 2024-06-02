from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOllama
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components_old.chat_callback_handler import ChatCallBackHandler
from components_old.common.langchain_component import embed_file
from enums.embedding_model import EmbeddingModel

import streamlit as st


@st.cache_resource()
def init_llm(param_model: str) -> ChatOllama:
    __model_name = EmbeddingModel(str(param_model)).model_full_name

    print('init_llm : ', __model_name)
    return ChatOllama(temperature=0.1, streaming=True, callbacks=[ChatCallBackHandler()], model=__model_name)


"""
    cache_resource start event 실행시 param 이 hash 가 불 가능한 type 인 경우 오류가 발생 한다
    이 경우 에는 param 앞에 _를 붙이면 된다 
"""


@st.cache_resource
def init_memory(_param: ChatOllama):
    return ConversationSummaryBufferMemory(
        llm=_param,
        max_token_limit=120,
        memory_key="chat_history",
        return_messages=True,
    )


# st.cache_data : 직렬화 가 가능한 값 (기본형 or 객체 구조를 가지는 반환값) 을 저장할 때 사용
# st.cache_resource : 직렬화 가 불가능한 값 (DB datasource, M/L Model ...) 을 저장할 때 사용
# function input param 이 변경될 때에만 다시 실행 된다
@st.cache_resource(show_spinner="Embedding file...")
def embed_file_wrapper(upload_file: UploadedFile, param_model: str):
    print('embed_file_wrapper : ', param_model)
    return embed_file(upload_file, './.cache/private_files', './.cache/private_embeddings',
                      EmbeddingModel(param_model))
