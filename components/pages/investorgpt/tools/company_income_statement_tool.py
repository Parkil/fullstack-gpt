import os
import requests
from typing import Type, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")


class CompanyIncomeStatementToolArgsSchema(BaseModel):
    symbol: str = Field(description="Stock Symbol of the company. Example: APPL, TSLA")


# 회사의 손익 계산서
class CompanyIncomeStatementTool(BaseTool):
    name = "CompanyIncomeStatementTool"
    description = """
    Use this to get the income statement of a company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyIncomeStatementToolArgsSchema] = CompanyIncomeStatementToolArgsSchema

    def _run(self, symbol) -> Any:
        result = requests.get(
            f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}')
        return result.json()['annualReports']
