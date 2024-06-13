import os
import requests
from typing import Type, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")


class CompanyTimeSeriesWeeklyToolArgsSchema(BaseModel):
    symbol: str = Field(description="Stock Symbol of the company. Example: APPL, TSLA")


# 주간 주가
class CompanyTimeSeriesWeeklyTool(BaseTool):
    name = "CompanyTimeSeriesWeeklyTool"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyTimeSeriesWeeklyToolArgsSchema] = CompanyTimeSeriesWeeklyToolArgsSchema

    def _run(self, symbol) -> Any:
        result = requests.get(
            f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}')
        org_json = result.json()['Weekly Time Series']
        return {k: v for k, v in org_json.items() if k >= '2024-03-22'}
