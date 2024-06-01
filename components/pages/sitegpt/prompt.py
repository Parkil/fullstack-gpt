from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def pick_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
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


def find_answer_prompt() -> ChatPromptTemplate:
    system_prompt: str = """
        Using ONLY the following context answer the user's question. If you can't just say you don't know,
        don't make anything up.
    
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
    """

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}"),
    ])

