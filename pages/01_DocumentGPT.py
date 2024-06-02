import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.langchain.callback_handler.streaming_chat_callback_handler import StreamingChatCallBackHandler
from components.langchain.file_parser import parse_by_file_embedding
from components.langchain.init_llm import initialize_open_ai_llm
from components.langchain.init_memory import initialize_conversation_memory
from components.pages.common.chat_message import print_message, print_message_history, print_message_and_save, \
    save_message, clear_message_history
from components.pages.documentgpt.prompt import find_document_gpt_prompt
from components.util.util import format_docs

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)


@st.cache_resource
def init_open_ai_streaming() -> ChatOpenAI:
    return initialize_open_ai_llm(streaming=True, callbacks=[StreamingChatCallBackHandler()])


@st.cache_resource
def init_open_ai() -> ChatOpenAI:
    return initialize_open_ai_llm()


# ConversationSummaryBufferMemory 에서 요약을 수행 하는 llm 설정에 callback 이 설정 되어 있을 경우
# 요약 정보를 얻기 위해 llm 을 수행 하면 callback 이 같이 작동 한다
@st.cache_resource
def init_memory() -> ConversationSummaryBufferMemory:
    return initialize_conversation_memory(chat_model=init_open_ai(),
                                          memory_key='chat_history', return_messages=True)


memory = init_memory()


# st.cache_data : 직렬화 가 가능한 값 (기본형 or 객체 구조를 가지는 반환값) 을 저장할 때 사용
# st.cache_resource : 직렬화 가 불가능한 값 (DB datasource, M/L Model ...) 을 저장할 때 사용
# function input param 이 변경될 때에만 다시 실행 된다
@st.cache_resource(show_spinner="Embedding file...")
def find_docs_by_file(upload_file: UploadedFile):
    return parse_by_file_embedding(upload_file, './.cache/files', './.cache/embeddings')


def __load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def __invoke_chain(chain_param, question):
    chain_result = chain_param.invoke(question)
    memory.save_context({"input": question}, {"output": chain_result.content})
    return chain_result


st.title('DocumentGPT')

st.markdown("""
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your files on sidebar
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])

# if file 자체가 일종의 react 의 state 처럼 작동 한다
message_group_key = 'document_gpt'
if file:
    retriever = find_docs_by_file(file)
    print_message('I`m ready! Ask Away', 'ai')
    print_message_history(message_group_key)

    message = st.chat_input("Ask anything about your file")

    if message:
        print_message_and_save(message_group_key, message, 'human')

        # chain 을 사용 하면
        # template.format_messages(context=docs, question=message) 나 retriever.invoke(message)
        # 같은 중간 단계 로직은 chain 이 실행 되는 과정 에서 자동 으로 실행 된다
        chain = ({
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                } | RunnablePassthrough.assign(chat_history=__load_memory)
                  | find_document_gpt_prompt() | init_open_ai_streaming())

        with st.chat_message('ai'):
            resp = __invoke_chain(chain, message)

        save_message(message_group_key, resp.content, 'ai')
else:
    clear_message_history(message_group_key)
