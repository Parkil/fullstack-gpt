from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from enums.embedding_model import EmbeddingModel


def embed_file(file: UploadedFile, file_dir: str, embedding_dir: str, model_type: EmbeddingModel) \
        -> VectorStoreRetriever:
    load_dotenv()
    docs = find_doc_list_from_file(file, file_dir)
    embeddings = __embedding_factory(model_type)

    # cache 에 데이터 가 없으면 OpenAI 를 통해서 embedding 하지만 cache 에 값이 있으면 cache 의 값을 불러 온다
    cache_dir = LocalFileStore(f"{embedding_dir}/{file.name}")
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    return vectorstore.as_retriever()


def find_doc_list_from_file(file: UploadedFile, file_dir: str) -> list[Document]:
    file_content = file.read()
    file_path = f"{file_dir}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100, separator="\n")
    loader = UnstructuredFileLoader(file_path)
    return loader.load_and_split(text_splitter=splitter)


def __embedding_factory(model_type: EmbeddingModel):
    if model_type == EmbeddingModel.OPEN_AI:
        return OpenAIEmbeddings()
    else:
        return OllamaEmbeddings(model=model_type.model_full_name)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
