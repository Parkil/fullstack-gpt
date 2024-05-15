import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from components.privategpt.streamlit_component import init_llm, init_memory, embed_file_wrapper
from components.privategpt.util import format_docs
from components.streamlit_component import init_session_singleton, send_message, paint_history

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)

st.title('PrivateGPT')

st.markdown("""
    Welcome!

    Use this chatbot to ask questions to an AI about your files!

    Upload your files on sidebar
""")

init_session_singleton('messages', [])

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


with st.sidebar:
    file = st.file_uploader("Upload a .txt, or .pdf or .docx file", type=['txt', 'pdf', 'docx'])

    if file:
        selected_model = st.selectbox(
            "Select Model",
            ("mistral", "wizardlm2", "falcon2"))

if file:
    llm = init_llm(selected_model)
    memory = init_memory(llm)

    retriever = embed_file_wrapper(file, str(selected_model))
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
