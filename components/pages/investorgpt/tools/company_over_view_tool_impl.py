import json
import os
from typing import Any

import requests
import yfinance as yf

from components.pages.investorgpt.tools.abstractclass.company_over_view_tool import CompanyOverViewTool


class AlphaVantageCompanyOverViewTool(CompanyOverViewTool):
    def _run(self, symbol: str) -> Any:
        alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        result = requests.get(
            f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}')
        return result.json()


class YahooFinanceCompanyOverViewTool(CompanyOverViewTool):
    def _run(self, symbol: str) -> Any:
        company = yf.Ticker(symbol)
        dict_ret = company.info
        return json.dumps(dict_ret, indent=4)
