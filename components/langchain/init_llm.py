from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI


# *[식별자], 또는 *, 파라메터, .. 는 전부 Named Parameter 로 강제 된다
# function_name(aaa, *bbb) -> bbb 는 Named Parameter 로 받아야 함
# function name(*, aaa, bbb, ccc, ddd) -> aaa,bbb,ccc,ddd 는 전부 Named Parameter 로 받아야 함

def initialize_open_ai_llm(*, temperature: float = 0.1, streaming: bool = False,
                           callbacks=None, model: str = 'gpt-3.5-turbo') -> ChatOpenAI:
    if callbacks is None:
        callbacks = []

    return ChatOpenAI(temperature=temperature, streaming=streaming, callbacks=callbacks, model=model)


def initialize_ollama_llm(*, temperature: float = 0.1, streaming: bool = False,
                          callbacks=None, model: str = 'mistral:latest'):
    if callbacks is None:
        callbacks = []

    return ChatOllama(temperature=temperature, streaming=streaming, callbacks=callbacks, model=model)

