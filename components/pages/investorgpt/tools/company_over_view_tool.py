import os

import requests
from typing import Type, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")


class CompanyOverViewToolArgsSchema(BaseModel):
    symbol: str = Field(description="Stock Symbol of the company. Example: APPL, TSLA")


# 회사의 요약 정보
class CompanyOverViewTool(BaseTool):
    name = "CompanyOverViewTool"
    description = """
    Use this to get an overview of the financials of the company.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyOverViewToolArgsSchema] = CompanyOverViewToolArgsSchema

    def _run(self, symbol) -> Any:
        result = requests.get(
            f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}')
        return result.json()
