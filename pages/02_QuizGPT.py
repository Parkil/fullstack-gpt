from typing import List

import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output
from langchain_openai import ChatOpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components_old.common.langchain_component import find_doc_list_from_file
from components_old.quizgpt.function_schema import quiz_function_schema
from components_old.quizgpt.langchain import wiki_search, invoke_question_function_chain

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title('QuizGPT')


@st.cache_resource
def init_llm():
    # functions 에 지정한 function schema 의 형식과 배치 되는 결과 형식을 prompt 에 입력할 경우 prompt 에 지정된 형식이 우선 한다
    # context window: llm prompt 에 들어 가는 token 의 크기 gpt-3.5-turbo-1106 의 경우 16,385 token 을 1개 prompt 에 보낼수 있다
    return (ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-1106", streaming=True,
                       callbacks=[StreamingStdOutCallbackHandler()])
            .bind(function_call="auto", functions=[quiz_function_schema()]))


llm = init_llm()


# streamlit cache 함수에 _를 붙이면 param hashing 을 통한 변경 감지가 안된다
# 즉 _를 붙인 파라메터만 있으면 계속 이전 데이터만 표시된다
# 궁금한점 : 동작 방식만 보면 st.cache_resource 가 invoke_chain_wrapper 함수가 호출 될때 이를 가로채서
# 파라메터 확인을 거친다음 invoke_chain_wrapper 함수를 호출한다는 이야기인데 파이썬에도 java reflection
# 같은 기능이 있나? 아니면 함수를 caching 하는 별도의 기능 or 라이브러리가 있나?
@st.cache_resource(show_spinner='Fetch Quiz Data...')
def generate_quiz(_docs_param: List[Document], _llm_param: ChatOpenAI | Runnable[Input, Output], topic_param: str):
    print(topic_param)
    return invoke_question_function_chain(_docs_param, _llm_param)


@st.cache_resource(show_spinner='Loading File...')
def generate_docs(upload_file: UploadedFile) -> List[Document]:
    return find_doc_list_from_file(upload_file, './.cache/quiz_files')


@st.cache_resource(show_spinner='Searching Wikipedia...')
def search_wikipedia(topic_param: str):
    return wiki_search(topic_param)


with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use", (
        "File", "Wikipedia Article",
    ))

    if choice == "File":
        file = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])
        if file:
            docs = generate_docs(file)
    else:
        topic = st.text_input("Search Wikipedia")

        if topic:
            docs = search_wikipedia(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )

else:
    # question_chain = {"context": format_docs} | question_prompt | llm
    # -> 결과 값을 context 에 할당
    # -> formatting_chain 실행
    #
    # RunnablePassthrough 와 {"context": question_chain} 의 차이
    # {"context": question_chain} 의 경우 LECL chain 에 맞는 type 을 넣는 책임이 개발자 에게 있음
    # RunnablePassthrough 의 경우 자동으로 LECL chain 에 맞게 변환이 되는 듯 함

    result = generate_quiz(docs, llm, topic if topic else file.name)

    with st.form("questions_form"):
        for question in result['questions']:
            st.write(question['question'])
            value = st.radio("Select an option", [answer['answer'] for answer in question['answers']], index=None)
            if {"answer": value, "correct": True} in question['answers']:
                st.success("Correct!")
            elif value is not None:
                st.error("Incorrect!")

        button = st.form_submit_button()
