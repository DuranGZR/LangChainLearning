from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import AliasChoices, BaseModel, Field


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

class GradeDocument(BaseModel):
    """
    Binery csore for revelance check on retrieved document.
    
    """

    binary_score : str = Field(
        validation_alias=AliasChoices("binary_score", "relevance", "score"),
        description="Document are revelant to the wuestion, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocument, method="json_mode")

system_promt = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. Respond in JSON."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_promt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader