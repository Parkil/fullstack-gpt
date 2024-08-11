from langchain_openai import ChatOpenAI
from crewai import Crew
from crewai.process import Process

from crew_ai_stock.agents import Agents
from crew_ai_stock.tasks import Tasks

agents = Agents()
tasks = Tasks()

researcher = agents.researcher()
technical_analyst = agents.technical_analyst()
financial_analyst = agents.financial_analyst()
hedge_fund_manager = agents.hedge_fund_manager()

research_task = tasks.research(researcher)
financial_task = tasks.finacial_analysis(financial_analyst)
technical_task = tasks.technical_analysis(technical_analyst)
recommend_task = tasks.investment_recommendation(hedge_fund_manager, [research_task, financial_task, technical_task])

crew = Crew(
    agents=[researcher, technical_analyst, financial_analyst, hedge_fund_manager, ],
    tasks=[research_task, financial_task, technical_task, recommend_task, ],
    verbose=True,
    process=Process.hierarchical,
    # Process.sequential (task 순차적 으로 실행), Process.hierarchical (task 계층적 으로 실행, manager_llm 옵션 필요)
    manager_llm=ChatOpenAI(model="gpt-4o"),
    memory=True,
)

result = crew.kickoff(inputs=dict(company="Salesforce"))
