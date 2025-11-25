from app.agent.state import AgentState
from app.agent.tools import retrieve_documents, get_stock_info
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from app.config import settings

# Initialize LLM with tools
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    model=settings.openrouter_model,
    temperature=0
)

from loguru import logger
logger.info(f"LLM Initialized with base_url: {llm.openai_api_base}, model: {llm.model_name}")

tools = [retrieve_documents, get_stock_info]
llm_with_tools = llm.bind_tools(tools)

def agent(state: AgentState):
    """
    Invokes the model with the current state (messages).
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ToolNode handles the execution of tools
tool_node = ToolNode(tools)
