import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.langchain.callback_handler.streaming_chat_callback_handler import StreamingChatCallBackHandler
from components.langchain.file_parser import parse_by_upload_file_and_disk_embedding
from components.langchain.init_llm import initialize_ollama_llm
from components.langchain.init_memory import initialize_conversation_memory
from components.pages.common.chat_message import print_message, print_message_history, print_message_and_save, \
    save_message, clear_message_history
from components.pages.privategpt.prompt import find_private_gpt_prompt
from components.util.util import format_docs
from enums.embedding_model import EmbeddingModel

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)

st.title('PrivateGPT')


@st.cache_resource
def init_ollama_ai_streaming(model_short_name: str) -> ChatOllama:
    print(f'init_ollama_ai_streaming(model_name : {model_short_name})')
    return initialize_ollama_llm(streaming=True, callbacks=[StreamingChatCallBackHandler()],
                                 model=EmbeddingModel(model_short_name).model_full_name)


@st.cache_resource
def init_ollama_ai(model_short_name: str) -> ChatOllama:
    print(f'init_ollama_ai(model_name : {model_short_name})')
    return initialize_ollama_llm(model=EmbeddingModel(model_short_name).model_full_name)


@st.cache_resource
def init_memory_local(model_short_name: str) -> ConversationSummaryBufferMemory:
    print(f'init_memory(model_name : {model_short_name})')
    llm_model = init_ollama_ai(model_short_name)
    return initialize_conversation_memory(chat_model=llm_model,
                                          memory_key='chat_history', return_messages=True)


# 다른 llm model 로 embedding 된 cache 를 가져 오려고 하면 AssertError 가 발생 한다
@st.cache_resource(show_spinner="Embedding file...")
def find_docs_by_file(upload_file: UploadedFile, model_short_name: str):
    return parse_by_upload_file_and_disk_embedding(upload_file, './.cache/private_files',
                         './.cache/private_embeddings', EmbeddingModel(model_short_name))


def __load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def __invoke_chain(chain_param, question):
    chain_result = chain_param.invoke(question)
    memory.save_context({"input": question}, {"output": chain_result.content})
    return chain_result


st.markdown("""
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your files on sidebar
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])

    if file:
        selected_model = st.selectbox(
            "Select Model",
            ("mistral", "wizardlm2", "falcon2"))

message_group_key = 'private_gpt'
if file:
    llm = init_ollama_ai_streaming(selected_model)
    memory = init_memory_local(selected_model)

    retriever = find_docs_by_file(file, str(selected_model))

    print_message('I`m ready! Ask Away', 'ai')
    print_message_history(message_group_key)

    message = st.chat_input("Ask anything about your file")

    if message:
        print_message_and_save(message_group_key, message, 'human')

        chain = {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                } | RunnablePassthrough.assign(chat_history=__load_memory) | find_private_gpt_prompt() | llm

        with st.chat_message('ai'):
            resp = __invoke_chain(chain, message)

        save_message(message_group_key, resp.content, 'ai')

else:
    clear_message_history(message_group_key)
