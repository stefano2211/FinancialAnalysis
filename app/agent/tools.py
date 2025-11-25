from langchain_core.tools import tool
from app.retrieval.searcher import SemanticSearcher
from app.agent.finance.graph import finance_graph
from typing import List

searcher = SemanticSearcher()

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve financial documents relevant to the query. 
    Use this tool when you need to find information in the documents to answer a user's question.
    """
    results = searcher.search(query, limit=5)
    
    formatted_docs = []
    for res in results:
        source = res["metadata"].get("source", "unknown")
        text = res["text"]
        formatted_docs.append(f"--- Documento: {source} ---\n{text}\n")
    
    return "\n".join(formatted_docs)

@tool
def get_stock_info(query: str) -> str:
    """
    Get stock information based on a user's query (e.g., "What is the price of Apple?", "Show me MSFT market cap").
    Use this tool when the user asks about stock prices or financial data for a specific company.
    """
    from langchain_core.messages import HumanMessage
    result = finance_graph.invoke({"messages": [HumanMessage(content=query)]})
    # The result from the subgraph will contain the stock_data in the state
    stock_data = result.get("stock_data", {})
    if not stock_data:
        return "No data found or could not identify ticker."
    return str(stock_data)
