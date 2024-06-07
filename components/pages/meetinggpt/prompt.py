from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def find_first_summary_prompt():
    return ChatPromptTemplate.from_template(
        """
        Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:                
    """
    )


def find_other_summary_prompt():
    return ChatPromptTemplate.from_template(
        """
        Your job is to produce a final summary.
        We have provided an existing summary up to a certain point: {existing_summary}
        We have the opportunity to refine the existing summary (only if needed) with some more context below.
        ------------
        {context}
        ------------
        Given the new context, refine the original summary.
        If the context isn't useful, RETURN the original summary.
        """
    )


def find_qna_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
                    Answer the question using ONLY the following context. 
                    If you don't know the answer just say you don't know. DON'T make anything up.

                    Context: {context}
                """),
        ("human", "{question}")
    ])


def find_qna_pick_answer_prompt() -> ChatPromptTemplate:
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


def find_qna_answer_list_prompt() -> ChatPromptTemplate:
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
        ("human", "Question: {question}"),
    ])
