from langchain.memory import ConversationSummaryBufferMemory


def initialize_conversation_memory(*, chat_model=None, max_token_limit=120,
                                   memory_key="history", return_messages=False) -> ConversationSummaryBufferMemory:
    return ConversationSummaryBufferMemory(
        llm=chat_model,
        max_token_limit=max_token_limit,
        memory_key=memory_key,
        return_messages=return_messages,
    )


