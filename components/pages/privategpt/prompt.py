from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def find_private_gpt_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
        Answer the question using ONLY the following context.
        If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
    """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
