from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)



system_promt = "Translate the following into {language}"
promt_tamplate = ChatPromptTemplate.from_messages(
    [
        ("system" , system_promt), ("user", "{text}")
    ]
)


parser = StrOutputParser()

chain = promt_tamplate | model | parser

app = FastAPI(
    title="Simple Messages API",
    description="An API that translates text into a specified language using a language model.",
    version="1.0.0"
)

add_routes(
    app, 
    chain,
    path= "/chain"
    )


if __name__ == "__main__":
    
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
    