"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, TypedDict, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, MessagesState

from pydantic import BaseModel


class Configuration(TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


class Article(BaseModel):
    title: str
    description: str


class AgentState(MessagesState):
    title: Optional[str] = None
    description: Optional[str] = None
    final_response: Optional[Article] = None


from langchain_core.tools import tool

@tool
def extract_title(text: str) -> str:
    """Extract a title from the given user query or content."""
    return "Dummy title"

@tool
def extract_description(text: str) -> str:
    """Extract a detailed description from the user query or content."""
    return "Dummy description"


from langchain.chat_models import init_chat_model

tools = [extract_title, extract_description]
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
model_with_tools = model.bind_tools(tools, tool_choice="any")

from langgraph.prebuilt import ToolNode
tool_node = ToolNode(tools)

def apply_tool_result(state: AgentState):
    last_msg = state["messages"][-1]
    print(last_msg)
    tool_call = last_msg.name
    result = state["messages"][-1].content  # the result from the tool call

    updated = {}
    if tool_call == "extract_title":
        updated["title"] = result
    elif tool_call == "extract_description":
        updated["description"] = result

    return updated


def construct_article(state: AgentState):
    if state["title"] and state["description"]:
        article = Article(title=state["title"], description=state["description"])
        return {"final_response": article}
    return {}



async def call_model(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    system_message = {"role": "system", "content": "You are a helpful assistant."}
    
    # Prepend the system message
    messages = [system_message] + state["messages"]

    response = await model_with_tools.ainvoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]


    # if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    #     return "tools"

    if not ("title" in state) or not ("description" in state):
        return "tools"  # ask to extract description
    elif state["title"] and state["description"]:
        return "finish"

    return "tools"


from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("apply_result", apply_tool_result)
workflow.add_node("finish", construct_article)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "finish": "finish"
    }
)

workflow.add_edge("tools", "apply_result")
workflow.add_edge("apply_result", "agent")
workflow.add_edge("finish", END)

graph = workflow.compile()
