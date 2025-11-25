from langgraph.graph import StateGraph, END
from app.agent.finance.state import FinanceState
from app.agent.finance.nodes import fetch_stock_data, finance_agent

# Define the graph
workflow = StateGraph(FinanceState)

# Add nodes
workflow.add_node("finance_agent", finance_agent)
workflow.add_node("fetch_stock_data", fetch_stock_data)

# Define edges
workflow.set_entry_point("finance_agent")
workflow.add_edge("finance_agent", "fetch_stock_data")
workflow.add_edge("fetch_stock_data", END)

# Compile the graph
finance_graph = workflow.compile()
