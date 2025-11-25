from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
import operator

class FinanceState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    ticker: str
    stock_data: dict
