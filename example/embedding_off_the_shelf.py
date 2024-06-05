from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

# from dotenv import load_dotenv
# import os
#
# load_dotenv()
#
# print(os.environ.get('OPENAI_API_KEY'))

llm = ChatOpenAI()

cache_dir = LocalFileStore("../.cache")

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

retriever=vectorstore.as_retriever()

# stuff: 여러 document 를 1개로 통합 해서 prompt 에 적재
# refine: document 를 순서 대로 summary 를 생성 하고 생성된 summary 를 이용 하여 다음 summary 를 생성 하는 식으로 loop 를 돈다
# ex) doc1 -> summary1 생성 -> doc2 + summary1 을 기반 으로 한 summary2 생성 -> doc3 + summary2 를 기반 으로 summary3 생성 .... 
# map reduce: 여러 document 마다 각각의 summary 를 생성한 다음 생성된 summaries 를 기반 으로 1개의 summary 를 생성
# map re-rank: document 나 llm 답변을 평가할 수 있게 하고 이중 평가가 가장 좋은 document 기반 답변 또는 llm 답변을 반환
embedd_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

result = embedd_chain.run("Describe Victory Mansions")

print(result)
