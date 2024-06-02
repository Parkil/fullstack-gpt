import streamlit as st
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from components.langchain.callback_handler.streaming_chat_callback_handler import StreamingChatCallBackHandler
from components.langchain.init_llm import initialize_open_ai_llm
from components.langchain.init_memory import initialize_conversation_memory
from components.langchain.site_parser import parse_by_sitemap_xml_embedding
from components.pages.common.chat_message import print_message, print_message_history, print_message_and_save, \
    save_message
from components.pages.common.session import set_session, get_session
from components.pages.sitegpt.event import on_button_click
from components.pages.sitegpt.prompt import find_answer_prompt, pick_answer_prompt

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


# streaming=False 로 설정시 StreamingChatCallBackHandler 의 on_llm_new_token event 는 작동 하지 않음
@st.cache_resource
def init_open_ai_streaming() -> ChatOpenAI:
    return initialize_open_ai_llm(streaming=True, callbacks=[StreamingChatCallBackHandler()])


@st.cache_resource
def init_open_ai() -> ChatOpenAI:
    return initialize_open_ai_llm()


non_streaming_llm = init_open_ai()
streaming_llm = init_open_ai_streaming()


@st.cache_resource(show_spinner="Fetching document by WebSite...")
def find_docs_by_sitemap(url_param: str, url_filter_param: str) -> VectorStoreRetriever:
    filter_arr = []

    if url_filter_param:
        filter_arr.append(fr"^(.*\/{url_filter_param}.*\/)")

    return parse_by_sitemap_xml_embedding(url_param, filter_arr)


@st.cache_resource
def init_memory() -> ConversationSummaryBufferMemory:
    return initialize_conversation_memory(chat_model=non_streaming_llm,
                                          memory_key='chat_history', return_messages=True)


memory = init_memory()


def __get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = RunnablePassthrough.assign(chat_history=__load_memory) | find_answer_prompt() | non_streaming_llm

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

    pick_chain = pick_answer_prompt() | streaming_llm

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
message_group_key = 'site_gpt'

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.google.com")
    url_filter = st.text_input("Write a filter string", placeholder="index")

    # 페이지 초기화 시 url 이 없을 경우 session_state 초기화
    if not url:
        set_session('run_search_sitemap', False)

    if st.button("Search SiteMap"):
        on_button_click(url)

if get_session('run_search_sitemap'):
    retriever = find_docs_by_sitemap(url, url_filter)

    if retriever is None:
        st.error("SiteMap Info Not Found")
        set_session('run_search_sitemap', False)
    else:
        message_group_key = f'${url}_{url_filter}'
        # text_input / chat_input 의 차이점
        # chat_input 은 전송을 하면 입력 값이 초기화 되지만 text_input 은 값이 초기화 안됨
        print_message('I`m ready! Ask Away', 'ai')
        print_message_history(message_group_key)

        query = st.chat_input("Ask question to the website")

        if query:
            print_message_and_save(message_group_key, query, 'human')

            # __get_answers 의 과정은 streaming 이 되어 서는 안되고 __pick_answer 의 과정만 streaming 이 되어야 한다
            # 그리고 추가 적으로 확인 해야 할 사항이 있는데 현재의 callback handler 설정 에서 memory 에 저장된 중간 과정이
            # streaming 되는 경우가 있는 듯 하다 추가 적인 확인 필요
            chain = ({"docs": retriever, "question": RunnablePassthrough()}
                     | RunnableLambda(__get_answers)
                     | RunnableLambda(__pick_answer))

            # StreamingChatCallBackHandler 에서 메시지 표시 하는 것을 담당
            with st.chat_message('ai'):
                resp = chain.invoke(query)

            # message_group_key 가 필요 하기 때문에 StreamingChatCallBackHandler on_llm_end 에서
            # save_message 를 호출 하는 것이 불가능
            save_message(message_group_key, resp.content, 'ai')



