import yfinance as yf
from app.agent.finance.state import FinanceState
from langchain_core.messages import ToolMessage
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from app.config import settings

# Initialize LLM for finance agent
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    model=settings.openrouter_model,
    temperature=0
)

def finance_agent(state: FinanceState):
    """
    Analyzes the query to extract the stock ticker.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [ToolMessage(content="Error: No messages provided.", tool_call_id="unknown")]}
    
    # Simple prompt to extract ticker
    system_prompt = SystemMessage(content="You are a financial assistant. Extract the stock ticker symbol from the user's query. Return ONLY the ticker symbol (e.g., AAPL, MSFT). If no ticker is found, return 'NONE'.")
    
    response = llm.invoke([system_prompt] + messages)
    ticker = response.content.strip().upper()
    
    if ticker == "NONE":
        return {"messages": [ToolMessage(content="Could not identify a ticker symbol.", tool_call_id="unknown")]}
        
    return {"ticker": ticker}

def fetch_stock_data(state: FinanceState):
    """
    Fetches stock data for the given ticker using yfinance.
    """
    ticker = state.get("ticker")
    if not ticker:
        return {"messages": [ToolMessage(content="Error: No ticker provided.", tool_call_id="unknown")]}
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant data to keep it concise
        data = {
            "symbol": info.get("symbol"),
            "shortName": info.get("shortName"),
            "currentPrice": info.get("currentPrice"),
            "marketCap": info.get("marketCap"),
            "currency": info.get("currency"),
            "recommendationKey": info.get("recommendationKey")
        }
        
        return {"stock_data": data, "messages": [ToolMessage(content=json.dumps(data), tool_call_id="unknown")]}
    except Exception as e:
        return {"messages": [ToolMessage(content=f"Error fetching data for {ticker}: {str(e)}", tool_call_id="unknown")]}
