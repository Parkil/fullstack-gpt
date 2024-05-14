import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.chat_callback_handler import ChatCallBackHandler
from components.langchain_component import embed_file
from components.streamlit_component import init_session_singleton, send_message, paint_history
from enums.embedding_model import EmbeddingModel

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)

init_session_singleton('messages', [])

llm = ChatOllama(temperature=0.1, streaming=True, callbacks=[ChatCallBackHandler()], model="mistral:latest")


@st.cache_resource
def init_memory():
    print('init_memory called')
    return ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=120,
        memory_key="chat_history",
        return_messages=True,
    )


# @st.cache_resource 처리시 inline 으로 직접 호출 하는 경우 오류가 발생 하는 경우가 있다
# 특히 함수 내부 에서 @st.cache_resource 에 지정된 함수를 직접 호출 하는 경우 그런듯
memory = init_memory()


# st.cache_data : 직렬화 가 가능한 값 (기본형 or 객체 구조를 가지는 반환값) 을 저장할 때 사용
# st.cache_resource : 직렬화 가 불가능한 값 (DB datasource, M/L Model ...) 을 저장할 때 사용
# function input param 이 변경될 때에만 다시 실행 된다
@st.cache_resource(show_spinner="Embedding file...")
def embed_file_wrapper(upload_file: UploadedFile):
    return embed_file(upload_file, './.cache/private_files', './.cache/private_embeddings', EmbeddingModel.OLLAMA)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Answer the question using ONLY the following context.
        If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


def load_memory(input_param):
    print(memory.load_memory_variables({}))
    print("""\n\n""")
    return memory.load_memory_variables({})["chat_history"]


def invoke_chain(question):
    chain_result = chain.invoke(question)
    memory.save_context({"input": question}, {"output": chain_result.content})
    return chain_result


st.title('PrivateGPT')

st.markdown("""
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your files on sidebar
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])

# if file 자체가 일종의 react 의 state 처럼 작동 한다
if file:
    retriever = embed_file_wrapper(file)
    send_message('I`m ready! Ask Away', 'ai', save=False)
    paint_history()

    message = st.chat_input("Ask anything about your file")

    if message:
        send_message(message, 'human')

        # chain 을 사용 하면
        # template.format_messages(context=docs, question=message) 나 retriever.invoke(message)
        # 같은 중간 단계 로직은 chain 이 실행 되는 과정 에서 자동 으로 실행 된다
        chain = {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                } | RunnablePassthrough.assign(chat_history=load_memory) | prompt | llm

        with st.chat_message('ai'):
            resp = invoke_chain(message)

else:
    st.session_state['messages'] = []
