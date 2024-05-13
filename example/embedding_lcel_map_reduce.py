from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

cache_dir = LocalFileStore("../.cache")

# chunk_size(문자를 분할할 때 크기), chunk_overlap(분할되는 이전/다음 chunk 의 데이터를 현재 데이터에 덧붙임, chunk 마다 중복되는 부분이 있을 수 있음)
# splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# token : llm model 에서 문자를 불규칙한 크기로 묶은 집합 ex) hello -> [he][llo] 이런 식으로 llm 에서 분류하고 [he][llo] 자체가 token 이 된다
# embed : 특정한 token 에 대해 n차원 (= n개의 특성) 의 각 차원 별로 평가 점수? ( = 특성에 얼마나 부합하는지 ) 를 부여 ( = vector 화 )
# token 별로 embed 작업을 통해 값이 부여 되면 이를 가지고 연산을 통해 새로운 값을 도출 하는 것이 가능

splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=100, separator="\n")

loader = UnstructuredFileLoader("../files/sample.txt")
docs = loader.load_and_split(text_splitter=splitter)
embeddings = OpenAIEmbeddings()

# cache 에 데이터 가 없으면 OpenAI 를 통해서 embedding 하지만 cache 에 값이 있으면 cache 의 값을 불러 온다
cache_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

vectorstore = FAISS.from_documents(docs, cache_embeddings)

retriever = vectorstore.as_retriever()

map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. 
            Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

map_doc_chain = map_doc_prompt | llm


def map_docs(inputs):
    documents = inputs['documents']
    question = inputs['question']

    return "\n\n".join(
        map_doc_chain.invoke({"context": doc.page_content, "question": question}).content for doc in documents)


map_chain = {"documents": retriever, "question": RunnablePassthrough()} | RunnableLambda(map_docs)

final_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
            Given the following extracted parts of a long document and a question, create a final answer. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.
            ------
            {context}
            """,),
    ("human", "{question}"),
])

map_reduce_chain = {"context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm

result = map_reduce_chain.invoke("Where does Winston go to work?")
print(result)
