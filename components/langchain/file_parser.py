from typing import List

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.common.file_util import save_file
from components.langchain.embedding import disk_caching_embedding
from enums.embedding_model import EmbeddingModel


def parse_by_file_embedding(file: UploadedFile, base_dir: str, cache_storage_dir: str,
                            embedding_model: EmbeddingModel = EmbeddingModel.OPEN_AI) -> VectorStoreRetriever:
    file_path = save_file(file, base_dir)
    docs = __gen_docs_from_file(file_path=file_path)
    return disk_caching_embedding(docs, cache_storage_dir, embedding_model).as_retriever()


def parse_by_file(file: UploadedFile, base_dir: str) -> List[Document]:
    file_path = save_file(file, base_dir)
    return __gen_docs_from_file(file_path=file_path)


def __gen_docs_from_file(*, file_path: str, chunk_size=600, chunk_overlap=100, separator='\n') -> list[Document]:
    splitter = (CharacterTextSplitter
                .from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator))
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)
