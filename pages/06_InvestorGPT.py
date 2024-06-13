import streamlit as st
from langchain.agents import AgentType, AgentExecutor, initialize_agent
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
    result = agent.invoke(company)

    st.write(result['output'].replace('$', '\\$'))
