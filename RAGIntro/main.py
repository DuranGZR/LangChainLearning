from dotenv import load_dotenv
from langsmith import Client
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import bs4
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/" ),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content", "post-title","post-header")
        )
    )
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 

splits = text_splitter.split_documents(docs)

vectostore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

retriever = vectostore.as_retriever()

#RAG Promt
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)





if __name__ == "__main__":
    for chunk in rag_chain.stream("What is maximum inner product search?"):
        print(chunk, end="", flush=True)