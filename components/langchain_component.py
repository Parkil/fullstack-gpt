from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import CharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile


def embed_file(file: UploadedFile, base_dir: str) -> VectorStoreRetriever:
    file_content = file.read()
    file_path = f"{base_dir}/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    load_dotenv()
    cache_dir = LocalFileStore(f"{base_dir}/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100, separator="\n")

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()

    # cache 에 데이터 가 없으면 OpenAI 를 통해서 embedding 하지만 cache 에 값이 있으면 cache 의 값을 불러 온다
    cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cache_embeddings)
    return vectorstore.as_retriever()
