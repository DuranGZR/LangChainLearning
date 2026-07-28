from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import AliasChoices, BaseModel, Field
from typing import Literal

class RouterQuery(BaseModel):
    """
    Route a user query to the most revelant datasource
    
    
    """
    
    source : Literal["vectorstore", "websearch"] = Field(
        ...,
        validation_alias=AliasChoices("source", "datasource"),
        description="Given a user question choose to route it to web search or vectorstore"
    )


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

structured_llm_router = llm.with_structured_output(RouterQuery, method="json_mode")

system_promt = """
You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.
Respond in JSON.
"""
route_promt = ChatPromptTemplate.from_messages([
    ("system", system_promt),
    ("human", "{question}")
])

question_router = route_promt | structured_llm_router