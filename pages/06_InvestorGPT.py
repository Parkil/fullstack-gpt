import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent, AgentType, initialize_agent
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from components.langchain.init_llm import initialize_open_ai_llm
from components.pages.investorgpt.tools.company_income_statement_tool import CompanyIncomeStatementTool
from components.pages.investorgpt.tools.company_over_view_tool import CompanyOverViewTool
from components.pages.investorgpt.tools.company_weekly_series_tool import CompanyTimeSeriesWeeklyTool
from components.pages.investorgpt.tools.stock_market_symbol_tool import StockMarketSymbolSearchTool


@st.cache_resource
def init_open_ai() -> ChatOpenAI:
    return initialize_open_ai_llm(model="gpt-3.5-turbo-1106")


@st.cache_resource
def init_agent() -> AgentExecutor:
    tools = [
        StockMarketSymbolSearchTool(),
        CompanyOverViewTool(),
        CompanyIncomeStatementTool(),
        CompanyTimeSeriesWeeklyTool(),
    ]

    # stock_agent = create_react_agent(llm=init_open_ai(), tools=tools, prompt=find_agent_prompt())
    # return AgentExecutor(agent=stock_agent, tools=tools, verbose=True)
    
    # initialize_agent 가 폐기되고 create_react_agent 를 쓰라고는 하는데 prompt 설정에 대한 정확한 설명이 안나와 있어서 그런가
    # create_react_agent 는 CompanyIncomeStatementTool, CompanyTimeSeriesWeeklyTool 이 작동이 안된다
    # 현재(2024-06-13) 까지의 정보로 추측하기로는 initialize_agent는 tool을 실행하는 기준을 langchain 로직으로 실행했는데 
    # react_agent는 tool 실행 기준을 LLM에 넘긴것으로 추정
    return initialize_agent(
        llm=init_open_ai(),
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[
            StockMarketSymbolSearchTool(),
            CompanyOverViewTool(),
            CompanyIncomeStatementTool(),
            CompanyTimeSeriesWeeklyTool(),
        ],
        agent_kwargs={
            "system_message": SystemMessage(content="""
            You are a hedge fund manager.

            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.

            Consider the performance of a stock, the company overview and the income statement.

            Be assertive in your judgement and recommend the stock or advise the user against it.
            """)
        }
    )


st.set_page_config(
    page_title="InvestorGPT",
    page_icon="💼",
)

st.markdown(
    """
    # InvestorGPT

    Welcome to InvestorGPT.

    Write down the name of a company and our Agent will do the research for you.
"""
)

company = st.text_input("Write a name of the company you are interested on")

if company:
    agent = init_agent()
    result = agent.invoke({"input": company})

    st.write(result['output'].replace('$', '\\$'))
