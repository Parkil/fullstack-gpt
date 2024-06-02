from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VST
from langchain_openai import OpenAIEmbeddings

from enums.embedding_model import EmbeddingModel


def in_memory_embedding(docs: list[Document], embedding_model: EmbeddingModel = EmbeddingModel.OPEN_AI) -> VST:
    return FAISS.from_documents(docs, __embedding_factory(embedding_model))


def disk_caching_embedding(docs: list[Document], cache_storage_dir: str,
                           embedding_model: EmbeddingModel = EmbeddingModel.OPEN_AI) -> VST:
    cache_storage = LocalFileStore(cache_storage_dir)
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(__embedding_factory(embedding_model), cache_storage)
    return FAISS.from_documents(docs, cache_embeddings)


def __embedding_factory(model_type: EmbeddingModel):
    if model_type == EmbeddingModel.OPEN_AI:
        return OpenAIEmbeddings()
    else:
        return OllamaEmbeddings(model=model_type.model_full_name)
