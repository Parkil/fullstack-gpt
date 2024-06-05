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
