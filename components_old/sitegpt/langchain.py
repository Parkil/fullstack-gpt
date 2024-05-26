import asyncio
from typing import List

from langchain_community.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


def async_chromium_loader(url: str):
    # Window 의 경우 아래 코드를 써주지 않으면 NotImplementedError 가 발생한다
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loader = AsyncChromiumLoader([url])
    docs = loader.load()

    html2text = Html2TextTransformer()
    return html2text.transform_documents(docs)


'''
실제로 돌려보니 sitemap.xml 이 있다고 해도 실제 fetch 가 되는 사이트 가 있고 아닌 사이트 가 있다
대표적으로 https://www.openai.com/sitemap.xml 은 sitemap 을 가져 오지 못한다
그리고 https://langchain.readthedocs.io/sitemap.xml 처럼 안에 있는 사이트 의 url 이 404를 표시 하는 경우도 있다
그리고 sitemap 의 url 에서 바로 내용을 보여 주지 않고 url -> 페이지 로드 -> redirect 를 한다면 sitemap 만 이용 해서
내용을 가져 온다는 것은 불가능 에 가깝다  

[테스트 시 사용할 URL]
https://pypi.org/e1.sitemap.xml
django-adminstats
'''


def sitemap_loader(url: str, filter_url_list: List[str]):
    print(url)
    # filter_urls : 지정된 URL 에 해당 되는 sitemap 만 가져 오도록 지정, url 명을 직접 적거나 r"^(.*\/Solution)" 처럼 정규식 을 사용 하는 것도 가능
    # restrict_to_same_domain: 만약 sitemap.xml 이 다른 domain 에 있는 경우 True 면 값을 가져 오지 못하도록 설정, False 면 값을 가져 오도록 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=200, separator='  ')
    loader = SitemapLoader(url, parsing_function=parse_page, restrict_to_same_domain=False, filter_urls=filter_url_list)
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


'''
SitemapLoader 에서 parsing 한 데이터 를 후처리 할 때 사용
다만 특정 사이트 에 맞춰서 개발을 하면 모를까 범용 으로 사용 하려먼 해당 방식은 맞지 않는다
'''


def parse_page(soup):
    header = soup.find("div", {"class": "framer-oqq5vb"})
    if header:
        header.decompose()

    return str(soup.get_text()).replace('\n', ' ').replace('\xa0', ' ')
