import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from components.langchain.init_llm import initialize_open_ai_llm
from components.langchain.init_memory import initialize_conversation_memory
from components.langchain.site_parser import parse_by_sitemap_xml
from components.pages.common.session import set_session
from components.pages.sitegpt.event import on_button_click
from components.pages.sitegpt.prompt import find_answer_prompt, pick_answer_prompt

# 수정사항
# 1. history 추가
# 2. memory 추가 - 처리완료
# 3. ui 를 document gpt 와 동일 하게 설정
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


@st.cache_resource(show_spinner="Initialize LLM..")
def init_open_ai() -> ChatOpenAI:
    return initialize_open_ai_llm()


llm = init_open_ai()


@st.cache_resource(show_spinner="Fetching document by WebSite...")
def find_docs_by_sitemap(url_param: str, url_filter_param: str) -> VectorStoreRetriever:
    filter_arr = []

    if url_filter_param:
        filter_arr.append(fr"^(.*\/{url_filter_param}.*\/)")

    return parse_by_sitemap_xml(url_param, filter_arr)


@st.cache_resource
def init_memory() -> ConversationSummaryBufferMemory:
    return initialize_conversation_memory(chat_model=llm, memory_key='chat_history', return_messages=True)


memory = init_memory()


def __get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = RunnablePassthrough.assign(chat_history=__load_memory) | find_answer_prompt() | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": __invoke_answer_chain(answers_chain, question, doc.page_content),
                "source": doc.metadata["source"],
                "date": 'None',
            } for doc in docs
        ]
    }


# _ -> 사용자 의 질문 + context
def __load_memory(_):
    return memory.load_memory_variables({})["chat_history"]


def __invoke_answer_chain(chain_param, question, context):
    chain_result = chain_param.invoke(
        {"question": question, "context": context}
    )
    memory.save_context({"input": question}, {"output": chain_result.content})
    return chain_result


def __pick_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    pick_chain = pick_answer_prompt() | llm

    condensed = "\n\n".join(f"Answer: {answer['answer']}\nSource: {answer['source']}\ndate: {answer['date']}\n"
                            for answer in answers)
    return pick_chain.invoke({
        "question": question,
        "answers": condensed
    })


st.title('SiteGPT')

st.markdown(
    """
    Ask Questions about the content of a website.
    
    Start by writing URL of the website on the sidebar
    """
)

# session state 는 페이지 를 닫지 않는 이상 유지 되는 값이기 때문에 현재 페이지 의 input value 를 기반 으로 한 session state 의 경우
# 현재 페이지 -> 다른 페이지 -> 현재 페이지 로 이동시 문제가 발생할 소지가 있다
# https://pypi.org/e1.sitemap.xml auto, wind, django
with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.google.com")
    url_filter = st.text_input("Write a filter string", placeholder="index")

    # 페이지 초기화 시 url 이 없을 경우 session_state 초기화
    if not url:
        set_session('run_search_sitemap', False)

    if st.button("Search SiteMap"):
        on_button_click(url)

if st.session_state.run_search_sitemap:
    retriever = find_docs_by_sitemap(url, url_filter)

    if retriever is None:
        st.error("SiteMap Info Not Found")
        set_session('run_search_sitemap', False)
    else:
        # text_input / chat_input 의 차이점
        # chat_input 은 전송을 하면 입력 값이 초기화 되지만 text_input 은 값이 초기화 안됨
        query = st.chat_input("Ask question to the website")

        if query:
            chain = ({"docs": retriever, "question": RunnablePassthrough()}
                     | RunnableLambda(__get_answers)
                     | RunnableLambda(__pick_answer))

            result = chain.invoke(query)
            st.write(result.content)

