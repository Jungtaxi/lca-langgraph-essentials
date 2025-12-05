# https://github.com/dzhng/deep-research
from dotenv import load_dotenv

load_dotenv()

import os
from typing import Literal
from tavily import TavilyClient
from deepagents import create_deep_agent

tavily_client = TavilyClient()

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

from langchain_upstage import ChatUpstage
model = ChatUpstage(model="solar-pro2")

agent = create_deep_agent(
    model=model,
    tools=[internet_search],
    system_prompt=research_instructions
)

# result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})

# # Print the agent's response
# print(result["messages"][-1].content)