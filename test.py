import streamlit as st
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from components_old.sitegpt.langchain import sitemap_loader

from langchain.prompts import ChatPromptTemplate

# 수정사항
# 1. history 추가
# 2. memory 추가
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


@st.cache_resource
def init_llm():
    # functions 에 지정한 function schema 의 형식과 배치 되는 결과 형식을 prompt 에 입력할 경우 prompt 에 지정된 형식이 우선 한다
    # context window: llm prompt 에 들어 가는 token 의 크기 gpt-3.5-turbo-1106 의 경우 16,385 token 을 1개 prompt 에 보낼수 있다
    return ChatOpenAI(temperature=0.1)


@st.cache_resource(show_spinner="Fetching document by WebSite...")
def find_docs(url_param: str):
    return sitemap_loader(url_param, [r"^(.*\/django.*\/)"])


llm = init_llm()

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ),
                "source": doc.metadata["source"],
                "date": 'None',
            } for doc in docs
        ]
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]
    print(question)
    print(answers)

    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(f"Answer: {answer['answer']}\nSource: {answer['source']}\ndate: {answer['date']}\n"
                            for answer in answers)
    return choose_chain.invoke({
        "question": question,
        "answers": condensed
    })


st.title('SiteGPT')

st.markdown(
    """
    Ask Questions about the content of a website.

    Start by writing URL of the website on the sidebar
    """
)

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://www.google.com")

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a SiteMap URL")
    else:
        retriever = find_docs(url)
        query = st.text_input("Ask question to the website")

        if query:
            chain = ({"docs": retriever, "question": RunnablePassthrough()} | RunnableLambda(get_answers)
                     | RunnableLambda(choose_answer))

            result = chain.invoke(query)
            st.write(result.content)
