from langchain_core.prompts import ChatPromptTemplate
from pydantic import AliasChoices, BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq


llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        validation_alias=AliasChoices("binary_score", "grounded", "score"),
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="json_mode")

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. Respond in JSON."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader