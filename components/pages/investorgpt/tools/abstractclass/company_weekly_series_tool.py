import os
from abc import ABC

import requests
from typing import Type, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")


class CompanyTimeSeriesWeeklyToolArgsSchema(BaseModel):
    symbol: str = Field(description="Stock Symbol of the company. Example: APPL, TSLA")


# 주간 주가
class CompanyTimeSeriesWeeklyTool(BaseTool, ABC):
    name = "CompanyTimeSeriesWeeklyTool"
    description = """
    Use this to get the weekly performance of a company stock.
    You should enter a stock symbol.
    """
    args_schema: Type[CompanyTimeSeriesWeeklyToolArgsSchema] = CompanyTimeSeriesWeeklyToolArgsSchema

