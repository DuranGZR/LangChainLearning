from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)


store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    
    return store[session_id]



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps answer questions."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


chain = prompt | model

config = { "configurable": { "session_id": "firstchat" } }

with_messages_history = RunnableWithMessageHistory(chain, get_session_history)




if __name__ == "__main__":
    
    while True:
        user_input = input(">")
        for r in with_messages_history.stream(
            [
                HumanMessage(user_input),
                
            ],
            config = config
        ):
            print(r.content, end = " ")
    