from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()

print(os.environ.get('OPENAI_API_KEY'))

llm = ChatOpenAI()

cache_dir = LocalFileStore("./.cache")

# chunk_size(문자를 분할할 때 크기), chunk_overlap(분할되는 이전/다음 chunk 의 데이터를 현재 데이터에 덧붙임, chunk 마다 중복되는 부분이 있을 수 있음)
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# token : llm model 에서 문자를 불규칙한 크기로 묶은 집합 ex) hello -> [he][llo] 이런 식으로 llm 에서 분류하고 [he][llo] 자체가 token 이 된다
# embed : 특정한 token 에 대해 n차원 (= n개의 특성) 의 각 차원 별로 평가 점수? ( = 특성에 얼마나 부합하는지 ) 를 부여 ( = vector 화 )
# token 별로 embed 작업을 통해 값이 부여 되면 이를 가지고 연산을 통해 새로운 값을 도출 하는 것이 가능

splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100, separator="\n")

loader = UnstructuredFileLoader("./files/sample.txt")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()

# cache 에 데이터 가 없으면 OpenAI 를 통해서 embedding 하지만 cache 에 값이 있으면 cache 의 값을 불러 온다
cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = Chroma.from_documents(docs, cache_embeddings)

# stuff: 검색된 모든 doc 을 합쳐서 prompt 에 입력
# refine: 검색된 doc 마다 prompt 를 던져서 답을 얻어 내는 방식
# map reduce: 검색된 doc 을 각각 요약 해서 prompt 에 입력
# map re-rank: 검색된 doc 마다 점수를 부여 해서(prompt 에 질문) 가장 높은 점수를 가진 doc 과 관련된 답변을 반환
embedd_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

result = embedd_chain.run("Describe Victory Mansions")

print(result)
