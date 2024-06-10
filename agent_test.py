import asyncio

from dotenv import load_dotenv
from langchain.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(temperature=0.1)
# llm = ChatOllama(temperature=0.1, model="falcon2:latest")


def plus(a, b):
    return a + b


tool = StructuredTool.from_function(
    func=plus,
    name="Sum-Calculator",
    description="Use this to perform sums of two numbers. This tool take two arguments, both  should be numbers.",
)

"""
llm_with_tools = llm.bind_tools([tool])  # bind_tools 설정 시 name 에 공백이 들어 가면 안됨

response = llm_with_tools.invoke([HumanMessage(content="what is result 3 + 5?")])
print(response.content)
print(response.tool_calls) # bind 된 function 이 작동 될만한 상황 에서만 작동 한다
"""


agent_executor = create_react_agent(llm, [tool])


"""
response = agent_executor.invoke({"messages": [HumanMessage(content="what is result 3 + 5?")]})

print(response["messages"])
"""

# agent 실행 관련 메시지 를 Streaming 방식 으로 표시
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="what is result 3 + 5?")]}
):
    print(chunk)
    print("----")


"""
async def sss():
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content="what is result 3 + 5?")]}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

asyncio.run(sss())
"""
