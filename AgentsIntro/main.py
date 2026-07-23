from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt  import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
memory = InMemorySaver()
search = TavilySearchResults(max_results=2)

tools = [search]



agent_executor = create_react_agent(llm, tools,checkpointer= memory)


config = {"configurable": {"thread_id": "abc123"}}


if __name__ == "__main__":
    while True:
        user_input = input("> ")
        
        for chunk in agent_executor.stream(
            {"messages" : [HumanMessage(content=user_input)]},
            config=config
        ):
            print(chunk, end="", flush=True)
            print("--")