import os
from typing import Any

import requests
import yfinance as yf

from components.pages.investorgpt.tools.abstractclass.company_income_statement_tool import CompanyIncomeStatementTool


class AlphaVantageCompanyIncomeStatementTool(CompanyIncomeStatementTool):

    def _run(self, symbol: str) -> Any:
        alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        result = requests.get(
            f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}')
        return result.json()['annualReports']


class YahooFinanceCompanyIncomeStatementTool(CompanyIncomeStatementTool):

    def _run(self, symbol: str) -> Any:
        company = yf.Ticker(symbol)
        pd_frame = company.income_stmt
        return pd_frame.to_json(
            orient='records',  # Output format: list of records
            date_format='iso',  # Use ISO date format
            double_precision=2,  # Precision of floating-point numbers
            force_ascii=False,  # Preserve non-ASCII characters
            date_unit='ms'  # Timestamp unit in milliseconds
        )
