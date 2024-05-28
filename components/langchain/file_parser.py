from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.common.file_util import save_file


def parse_by_file(file: UploadedFile, base_dir: str) -> list[Document]:
    file_path = save_file(file, base_dir)
    return __gen_docs_from_file(file_path=file_path)


def __gen_docs_from_file(*, file_path: str, chunk_size=600, chunk_overlap=100, separator='\n') -> list[Document]:
    splitter = (CharacterTextSplitter
                .from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator))
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)
