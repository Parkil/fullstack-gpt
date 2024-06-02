import json
from typing import List

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from components.pages.quizgpt.json_output_parser import JsonOutputParser
from components.pages.quizgpt.prompt import find_question_prompt, find_formatting_prompt, find_question_function_prompt
from components.types.user_define_type import BindingLLMType
from components.util.util import format_docs


def invoke_question_format_chain(docs: List[Document], llm: ChatOpenAI):
    question_prompt = find_question_prompt()

    question_chain = {"context": format_docs} | question_prompt | llm

    formatting_prompt = find_formatting_prompt()

    formatting_chain = formatting_prompt | llm

    chain = {"context": question_chain} | formatting_chain | JsonOutputParser()
    return chain.invoke(docs)


# function call 방식 으로 chain 을 호출할 경우 Parser 를 사용할 수 없다
def invoke_question_function_chain(docs: List[Document], llm: BindingLLMType):
    question_prompt = find_question_function_prompt()

    chain = {"context": format_docs} | question_prompt | llm

    chain_result = chain.invoke(docs)

    json_formatted_str = chain_result.additional_kwargs['function_call']['arguments']
    return json.loads(json_formatted_str)
