import json
from typing import List

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from components.langchain_component import format_docs
from components.quizgpt.json_output_parser import JsonOutputParser
from components.quizgpt.prompt import find_question_prompt, find_formatting_prompt, find_question_function_prompt


def invoke_question_format_chain(docs: List[Document], llm: ChatOpenAI):
    question_prompt = find_question_prompt()

    question_chain = {"context": format_docs} | question_prompt | llm

    formatting_prompt = find_formatting_prompt()

    formatting_chain = formatting_prompt | llm

    chain = {"context": question_chain} | formatting_chain | JsonOutputParser()
    return chain.invoke(docs)


# function call 방식 으로 chain 을 호출할 경우 Parser 를 사용할수 없다
def invoke_question_function_chain(docs: List[Document], llm: ChatOpenAI):
    question_prompt = find_question_function_prompt()

    chain = {"context": format_docs} | question_prompt | llm

    chain_result = chain.invoke(docs)

    json_formatted_str = chain_result.additional_kwargs['function_call']['arguments']
    return json.loads(json_formatted_str)


def wiki_search(topic: str):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(topic)
