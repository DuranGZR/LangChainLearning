from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vector_store = Chroma.from_documents(
    documents = documents,
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

reteiever = vector_store.as_retriever(search_kwargs={"k": 1})


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

message = """
Answer this question using the  provided context only. 
{question}

Context:{context}

"""

promt = ChatPromptTemplate.from_messages(
    [
        
        ("human", message)
    ]
)


chain = {"context": reteiever , "question": RunnablePassthrough()} | promt | llm

if __name__ == "__main__":
    
    response = chain.invoke("tell me about dogs")
    print(response.content)