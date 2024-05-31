from langchain.memory import ConversationSummaryBufferMemory


def initialize_conversation_memory(*, chat_model, max_token_limit=120,
                                   memory_key="history", return_messages=False) -> ConversationSummaryBufferMemory:
    # 주의 사항 memory 의 내용을 chain 에 적용하고자 한다면 return_messages 는 반드시 True 로 설정이 되어야 한다
    return ConversationSummaryBufferMemory(
        llm=chat_model,
        max_token_limit=max_token_limit,
        memory_key=memory_key,
        return_messages=return_messages,
    )


