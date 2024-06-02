import asyncio
from typing import List, Sequence

from langchain_community.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter

from components.langchain.embedding import in_memory_embedding


def parse_by_wiki_topic(topic: str, fetch_count: int = 5) -> List[Document]:
    retriever = WikipediaRetriever(top_k_results=fetch_count)
    return retriever.get_relevant_documents(topic)


def parse_by_url(url: str) -> Sequence[Document]:
    # Window 의 경우 아래 코드를 써주지 않으면 NotImplementedError 가 발생 한다
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    loader = AsyncChromiumLoader([url])
    docs = loader.load()

    html2text = Html2TextTransformer()
    return html2text.transform_documents(docs)


def parse_by_sitemap_xml_embedding(site_map_xml_url: str, site_map_filter_urls: List[str])\
        -> VectorStoreRetriever | None:
    # filter_urls : 지정된 URL 에 해당 되는 sitemap 만 가져 오도록 지정, url 명을 직접 적거나 r"^(.*\/Solution)" 처럼 정규식 을 사용 하는 것도 가능
    # restrict_to_same_domain: 만약 sitemap.xml 이 다른 domain 에 있는 경우 True 면 값을 가져 오지 못하 도록 설정, False 면 값을 가져 오도록 설정
    loader = SitemapLoader(site_map_xml_url, parsing_function=__parse_page,
                           restrict_to_same_domain=False, filter_urls=site_map_filter_urls)
    loader.requests_per_second = 1
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=200, separator='  ')
    docs = loader.load_and_split(text_splitter=splitter)

    if len(docs) == 0:
        return None

    return in_memory_embedding(docs).as_retriever()


def __parse_page(soup):
    header = soup.find("div", {"class": "framer-oqq5vb"})
    if header:
        header.decompose()

    return str(soup.get_text()).replace('\n', ' ').replace('\xa0', ' ')
