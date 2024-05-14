import time

from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(temperature=0.1)

# memory_key : template 에 입력될 memory 정보를 mapping 하는 key
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=120,
    memory_key="chat_history",
    return_messages=True,
)

# MessageHolder: 위 template 의 {chat_history} 와 동일한 역할을 수행
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful AI taking to a human"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def load_memory(input_param):
    print(memory.load_memory_variables({}))
    print("""\n\n""")
    return memory.load_memory_variables({})["chat_history"]


# RunnablePassthrough : chain 이 실행될 때 chain 에 들어 가는 변수중 일부를 자동 할당
chain = RunnablePassthrough.assign(chat_history=load_memory) | chat_prompt_template | llm


def invoke_chain(question):
    chain_result = chain.invoke({
        "question": question,
    })

    memory.save_context({"input": question}, {"output": chain_result.content})

    print('chain_result : ', chain_result)
    print("""\n\n""")
    return chain_result


invoke_chain("My name is Test")
time.sleep(2)
invoke_chain("Who are you?")
time.sleep(2)
invoke_chain("What are the pokemons?")
time.sleep(2)
invoke_chain("What are the pikachu?")

