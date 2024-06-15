from components.pages.investorgpt.tools.company_income_statement_tool_impl import \
    AlphaVantageCompanyIncomeStatementTool, YahooFinanceCompanyIncomeStatementTool
from components.pages.investorgpt.tools.company_over_view_tool_impl import AlphaVantageCompanyOverViewTool, \
    YahooFinanceCompanyOverViewTool
from components.pages.investorgpt.tools.company_weekly_series_tool_impl import AlphaVantageTimeSeriesWeeklyTool, \
    YahooFinanceTimeSeriesWeeklyTool
from components.pages.investorgpt.tools.stock_market_symbol_tool_impl import DuckDuckGoSearchSymbolTool

ALPHA_VANTAGE_TOOLS = [
    DuckDuckGoSearchSymbolTool(),
    AlphaVantageCompanyOverViewTool(),
    AlphaVantageCompanyIncomeStatementTool(),
    AlphaVantageTimeSeriesWeeklyTool(),
]

YAHOO_FINANCE_TOOLS = [
    DuckDuckGoSearchSymbolTool(),
    YahooFinanceCompanyOverViewTool(),
    YahooFinanceCompanyIncomeStatementTool(),
    YahooFinanceTimeSeriesWeeklyTool(),
]
