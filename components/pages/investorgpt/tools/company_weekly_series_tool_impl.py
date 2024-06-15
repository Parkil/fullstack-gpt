import os
from typing import Any

import requests
import yfinance as yf

from components.pages.investorgpt.tools.abstractclass.company_weekly_series_tool import CompanyTimeSeriesWeeklyTool


class AlphaVantageTimeSeriesWeeklyTool(CompanyTimeSeriesWeeklyTool):

    def _run(self, symbol: str) -> Any:
        alpha_vantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        result = requests.get(
            f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}'
            f'&apikey={alpha_vantage_api_key}')
        org_json = result.json()['Weekly Time Series']
        return {k: v for k, v in org_json.items() if k >= '2024-03-22'}


class YahooFinanceTimeSeriesWeeklyTool(CompanyTimeSeriesWeeklyTool):

    def _run(self, symbol: str) -> Any:
        company = yf.Ticker(symbol)
        pd_frame = company.history(period="1y", interval="1wk")
        return pd_frame.to_json(
            orient='records',  # Output format: list of records
            date_format='iso',  # Use ISO date format
            double_precision=2,  # Precision of floating-point numbers
            force_ascii=False,  # Preserve non-ASCII characters
            date_unit='ms'  # Timestamp unit in milliseconds
        )
