from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

from components.langchain.embedding import in_memory_embedding
from components.langchain.init_llm import initialize_ollama_llm
from enums.embedding_model import EmbeddingModel

'''
Openai function agent 도 langsmith 를 보면 지정된 tool 을 돌려서 데이터 를 가져온 다음 이를 
최종 prompt 에서 합쳐서 보내는 방식을 사용 하고 있음 단 항목이 복잡 하게 설정 되어 있는 것으로 보아
Openai model 에서 별도의 prompt 형식이 따로 존재 하는 듯
'''

load_dotenv()

text_loader = TextLoader('./stockmarketsymbol.txt', encoding='utf-8')
docs1 = text_loader.load()
embeddings_stock_market_symbol = in_memory_embedding(docs1, EmbeddingModel.OLLAMA_WIZARDLM2)

json_loader1 = TextLoader('./company_overview.json')
docs2 = json_loader1.load()
embeddings_company_overview = in_memory_embedding(docs2, EmbeddingModel.OLLAMA_WIZARDLM2)

json_loader2 = TextLoader('./weekly_series.json')
docs3 = json_loader2.load()
embeddings_weekly_series = in_memory_embedding(docs3, EmbeddingModel.OLLAMA_WIZARDLM2)

retriever_1 = embeddings_stock_market_symbol.as_retriever()
retriever_2 = embeddings_company_overview.as_retriever()
retriever_3 = embeddings_weekly_series.as_retriever()

# 회사 정보를 가지고 주가 예측을 하는 데에는 wizardlm 이 나은듯
prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a hedge fund manager.

                You evaluate a company and provide your opinion and reasons why the stock is a buy or not.

                Consider the performance of a stock, the company overview and the income statement.

                Be assertive in your judgement and recommend the stock or advise the user against it.
                
                symbol_data: {symbol_data}
                company_overview_data: {company_overview_data}
                weekly_series_data: {weekly_series_data}
                """,
            ),
            ("human", "{companyName}"),
        ]
    )

llm = initialize_ollama_llm(model='wizardlm2:latest')

chain = prompt | llm

result = chain.invoke({
    "symbol_data": retriever_1,
    "company_overview_data": retriever_2,
    "weekly_series_data": retriever_3,
    "companyName": "Data Dog"
})

print(result.content)

