from typing import Any

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from components.pages.investorgpt.tools.abstractclass.stock_market_symbol_tool import StockMarketSymbolSearchTool


class DuckDuckGoSearchSymbolTool(StockMarketSymbolSearchTool):
    def _run(self, query: str) -> Any:
        ddg = DuckDuckGoSearchAPIWrapper()
        return ddg.run(query)
