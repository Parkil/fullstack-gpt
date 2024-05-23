import streamlit as st
from langchain_openai import ChatOpenAI

from components.sitegpt.langchain import sitemap_loader

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


@st.cache_resource
def init_llm():
    # functions 에 지정한 function schema 의 형식과 배치 되는 결과 형식을 prompt 에 입력할 경우 prompt 에 지정된 형식이 우선 한다
    # context window: llm prompt 에 들어 가는 token 의 크기 gpt-3.5-turbo-1106 의 경우 16,385 token 을 1개 prompt 에 보낼수 있다
    return ChatOpenAI(temperature=0.1)


@st.cache_resource(show_spinner="Fetching document by WebSite...")
def find_docs(url_param: str):
    return sitemap_loader(url_param, [r"^(.*\/django.*\/)"])


llm = init_llm()

st.title('SiteGPT')

st.markdown(
    """
    Ask Questions about the content of a website.
    
    Start by writing URL of the website on the sidebar
    """
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.google.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a SiteMap URL")
    else:
        docs = find_docs(url)
        st.write(docs)
