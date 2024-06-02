from enum import Enum


# 주의할 점 : Enum 에 OPEN_AI = 'openai', 처럼 , 를 붙이면 EmbeddingModel('openai') 호출 시 오류가 발생 한다

class EmbeddingModel(Enum):
    OPEN_AI = ('openai', '')
    OLLAMA_MISTRAL = ('mistral', 'mistral:latest')
    OLLAMA_FALCON2 = ('falcon2', 'falcon2:latest')
    OLLAMA_WIZARDLM2 = ('wizardlm2', 'wizardlm2:latest')

    def __new__(cls, model_short_name, model_full_name):
        obj = object.__new__(cls)
        obj._value_ = model_short_name
        obj.model_full_name = model_full_name
        return obj
