import streamlit as st
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from streamlit.runtime.uploaded_file_manager import UploadedFile

from components.chat_callback_handler import ChatCallBackHandler
from components.langchain_component import embed_file
from components.streamlit_component import init_session_singleton, send_message, paint_history
from enums.embedding_model import EmbeddingModel

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

init_session_singleton('messages', [])

llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallBackHandler()])

# 정상적인 케이스 Data {'chat_history': [SystemMessage(content='The human asks who Winston is, and the AI explains that
# Winston is a character in George Orwell\'s novel "1984." Winston is a thirty-nine-year-old man who works at the
# Records Department in the Ministry of Truth, altering historical records to fit the propaganda needs of the
# totalitarian regime led by Big Brother. Winston lives in Victory Mansions, a run-down building with issues like a
# non-functioning lift due to electricity cuts. The human then asks who Big Brother is, and the AI describes Big
# Brother as the omnipresent authoritarian leader of the regime, used as a propaganda tool to maintain control
# through fear and manipulation.')]}
#
# 비 정상적인 케이스 data {'chat_history': [HumanMessage(content='who is winston?'), AIMessage(content='Winston is a
# character in the context provided. He is described as a man who is thirty-nine years old, with fair hair,
# a sanguine face, and a meagre body. He lives in Victory Mansions and works at the Records Department. Winston is
# shown to have a sense of uneasiness and fear, especially when encountering certain individuals.')]}
#
# ollama mistral 에서는 비 정상적인 케이스가 나온적이 없는데 open_ai 에서는 비정상적인 케이스가 종종 발견된다 원인은 아직 파악이
# 안되었음
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


@st.cache_resource
def test():
    return ConversationChain(
        llm=llm,
        # We set a very low max_token_limit for the purposes of testing.
        memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=40),
        verbose=True,
    )


conversation_with_summary = test()


# st.cache_data : 직렬화 가 가능한 값 (기본형 or 객체 구조를 가지는 반환값) 을 저장할 때 사용
# st.cache_resource : 직렬화 가 불가능한 값 (DB datasource, M/L Model ...) 을 저장할 때 사용
# function input param 이 변경될 때에만 다시 실행 된다
@st.cache_resource(show_spinner="Embedding file...")
def embed_file_wrapper(upload_file: UploadedFile):
    return embed_file(upload_file, './.cache/files', './.cache/embeddings', EmbeddingModel.OPEN_AI)


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


st.title('DocumentGPT')

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
            # resp = conversation_with_summary.predict(input=message)
            resp = invoke_chain(message)

else:
    st.session_state['messages'] = []
