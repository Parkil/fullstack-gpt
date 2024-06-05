from typing import List

import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.langchain.file_parser import parse_by_upload_file
from components.langchain.init_llm import initialize_open_ai_binding_llm
from components.langchain.site_parser import parse_by_wiki_topic
from components.pages.quizgpt.chain import invoke_question_function_chain
from components.pages.quizgpt.function_schema import quiz_function_schema
from components.types.user_define_type import BindingLLMType

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title('QuizGPT')


@st.cache_resource
def init_open_ai_streaming() -> BindingLLMType:
    return initialize_open_ai_binding_llm(temperature=0.1, model="gpt-3.5-turbo-1106",
                                          streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
                                          functions=[quiz_function_schema()])


@st.cache_resource(show_spinner='Searching Wikipedia...')
def gen_docs_by_wiki(topic_param: str) -> List[Document]:
    return parse_by_wiki_topic(topic_param)


@st.cache_resource(show_spinner='Loading File...')
def gen_docs_by_file(upload_file: UploadedFile) -> List[Document]:
    return parse_by_upload_file(upload_file, './.cache/quiz_files')


@st.cache_resource(show_spinner='Generate Docs...')
def generate_docs(topic: UploadedFile | str) -> List[Document]:
    if isinstance(topic, str):
        return parse_by_wiki_topic(topic)
    else:
        return gen_docs_by_file(topic)


# streamlit cache 함수에 _를 붙이면 param hashing 을 통한 변경 감지가 안된다
# 즉 _를 붙인 파라메터만 있으면 계속 이전 데이터만 표시된다
# 궁금한점 : 동작 방식만 보면 st.cache_resource 가 invoke_chain_wrapper 함수가 호출 될때 이를 가로채서
# 파라메터 확인을 거친다음 invoke_chain_wrapper 함수를 호출한다는 이야기인데 파이썬에도 java reflection
# 같은 기능이 있나? 아니면 함수를 caching 하는 별도의 기능 or 라이브러리가 있나?
@st.cache_resource(show_spinner='Fetch Quiz Data...')
def generate_quiz(topic_param: UploadedFile | str):
    __llm = init_open_ai_streaming()
    __docs = generate_docs(topic_param)
    return invoke_question_function_chain(__docs, __llm)


chosen_topic: UploadedFile | str
docs: List[Document] = []
with st.sidebar:
    choice = st.selectbox("Choose what you want to use", (
        "File", "Wikipedia Article",
    ))

    if choice == "File":
        chosen_topic = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])
    else:
        chosen_topic = st.text_input("Search Wikipedia")


if not chosen_topic:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )

else:
    result = generate_quiz(chosen_topic)

    with st.form("questions_form"):
        for question in result['questions']:
            st.write(question['question'])
            value = st.radio("Select an option", [answer['answer'] for answer in question['answers']], index=None)
            if {"answer": value, "correct": True} in question['answers']:
                st.success("Correct!")
            elif value is not None:
                st.error("Incorrect!")

        button = st.form_submit_button()
