from app.agent.state import AgentState
from app.agent.tools import retrieve_documents, get_stock_info
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
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
    
    # System prompt for the main orchestrator
    system_prompt = SystemMessage(content="""You are a financial analysis orchestrator. You have access to two tools:
1. `retrieve_documents`: Use this to find information in internal financial documents (PDFs, reports).
2. `get_stock_info`: Use this to fetch real-time stock market data (prices, market cap, etc.).

Decide which tool to use based on the user's query:
- If the user asks for current market data, price, or stock performance, use `get_stock_info`.
- If the user asks about specific internal reports, strategic plans, or document content, use `retrieve_documents`.

You are responsible for analyzing the data returned by these tools to answer the user's question comprehensively.""")
    
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

# ToolNode handles the execution of tools
tool_node = ToolNode(tools)
