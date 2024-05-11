
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

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

vectorstore = FAISS.from_documents(docs, cache_embeddings)

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions using only the following context. "
               "If you don`t know the answer just say you don`t know, don`t make it up:\n\n{context}"),
    ("human", "{question}")
])

# RunnablePassthrough: 이전에는 prop 자동 할당 기능을 하는 걸로 알았는데 이제 보니 chain 에 값을 전달해주는 기능을 함
lcel_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

result = lcel_chain.invoke("What is Victory Mansions?")
print(result)

