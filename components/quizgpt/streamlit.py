from typing import List

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from components.langchain_component import format_docs
from components.quizgpt.langchain_parser import JsonOutputParser
from components.quizgpt.prompt import find_question_prompt, find_formatting_prompt


# 여기 서는 함수만 생성 하고 streamlit에서 @st.cache_resource 를 선언하고 해당 함수를 호출하는 것이 맞아보인다
# cache 를 선언하는건 엄연히 streamlit 이기 때문
def invoke_chain(docs: List[Document], llm: ChatOpenAI):
    question_prompt = find_question_prompt()

    question_chain = {"context": format_docs} | question_prompt | llm

    formatting_prompt = find_formatting_prompt()

    formatting_chain = formatting_prompt | llm

    chain = {"context": question_chain} | formatting_chain | JsonOutputParser()
    return chain.invoke(docs)


def wiki_search(topic: str):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)
