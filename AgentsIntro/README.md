# Agents Intro (LangChain ReAct Agent)

*Read this document in [English](#english-version) or [Turkish](#turkish-version).*

---

<a id="english-version"></a>
## 🇬🇧 English Version

### Project Overview
This project introduces a **LangChain agent** built with the **ReAct** pattern. Instead of answering with a single prompt only, the agent can decide when to use tools, call a web search tool, and then generate a final answer based on the retrieved information.

The current `main.py` demonstrates a simple interactive console chatbot with memory. It loads environment variables, initializes a Groq chat model, adds Tavily search as a tool, keeps conversation state with an in-memory checkpointer, and then runs in a loop so the user can ask questions repeatedly.

### Code-by-Code Explanation

#### 1. Imports and Environment Setup
```python
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
```
* `load_dotenv()` reads API keys from the local `.env` file so secrets stay out of the code.
* `ChatGroq` connects the app to Groq's hosted LLM inference.
* `TavilySearchResults` gives the agent a web search tool.
* `create_react_agent` builds the ReAct agent workflow.
* `InMemorySaver` stores short-term conversation state while the app is running.

#### 2. Model Initialization
```python
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
```
* This creates the chat model used by the agent.
* `temperature=0.2` keeps answers more stable and factual.

#### 3. Tool Setup
```python
search = TavilySearchResults(max_results=2)
tools = [search]
```
* `TavilySearchResults` lets the agent search the web when it needs outside information.
* `max_results=2` keeps the search output short and focused.
* `tools` is the list passed into the agent.

#### 4. Memory / Checkpoint Setup
```python
memory = InMemorySaver()
```
* This stores the conversation state in RAM.
* It is useful for keeping a conversation consistent within the current session.
* Because it is in-memory, the history is lost when the program stops.

#### 5. Agent Creation
```python
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
```
* This is the core line that turns the LLM plus tools into an agent.
* The agent can decide whether to answer directly or call the search tool first.
* `checkpointer=memory` connects the conversation history to a thread id.

#### 6. Session Configuration
```python
config = {"configurable": {"thread_id": "abc123"}}
```
* `thread_id` identifies the conversation thread.
* If you keep the same thread id, the agent can continue from the same chat state.

#### 7. Interactive Loop and Streaming Output
```python
if __name__ == "__main__":
    while True:
        user_input = input("> ")

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        ):
            print(chunk, end="", flush=True)
            print("--")
```
* `while True` keeps the chatbot running until you stop it manually.
* `HumanMessage(content=user_input)` wraps the user question in LangChain's message format.
* `.stream()` prints the response step by step instead of waiting for the whole answer.
* This gives a more live, chat-like console experience.

### What `main.py` Does
In short, `main.py` is the entry point of the agent app. It:

1. loads environment variables,
2. creates a Groq LLM,
3. adds Tavily web search as a tool,
4. keeps short-term memory with `InMemorySaver`,
5. builds a ReAct agent,
6. and runs an interactive chat loop in the terminal.

### Key Takeaways
* This example shows how to build a tool-using agent instead of a plain chatbot.
* The agent can search the web before answering.
* Memory works only during the current run because `InMemorySaver` is temporary.
* The `thread_id` is important because it separates one conversation from another.

---

<a id="turkish-version"></a>
## 🇹🇷 Türkçe Versiyon

### Proje Özeti
Bu proje, **LangChain ReAct Agent** yapısını tanıtan bir örnektir. Normal bir LLM sadece tek bir prompt ile cevap verirken, bu yapı gerektiğinde araç (tool) kullanabilir, web araması yapabilir ve ardından bulduğu bilgiye göre son cevabı oluşturabilir.

Bu klasördeki `main.py`, hafızalı ve terminalden çalışan basit bir sohbet uygulamasını gösterir. Ortam değişkenlerini yükler, Groq modelini başlatır, Tavily arama aracını ekler, oturum hafızasını `InMemorySaver` ile tutar ve kullanıcı soru sordukça döngü halinde çalışır.

### Kod Analizi: Neyin Neden Yapıldığı

#### 1. İçe Aktarımlar ve Ortam Yükleme
```python
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
```
* `load_dotenv()` yerel `.env` dosyasındaki API anahtarlarını okur.
* `ChatGroq`, uygulamayı Groq üzerindeki dil modeline bağlar.
* `TavilySearchResults`, agente web araması yapma yeteneği kazandırır.
* `create_react_agent`, araç kullanan ReAct agent akışını kurar.
* `InMemorySaver`, konuşma durumunu geçici olarak bellekte saklar.

#### 2. Model Başlatma
```python
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
```
* Burada kullanılacak sohbet modeli tanımlanır.
* `temperature=0.2`, cevapların daha kontrollü ve tutarlı olmasını sağlar.

#### 3. Araçların Hazırlanması
```python
search = TavilySearchResults(max_results=2)
tools = [search]
```
* `TavilySearchResults`, gerektiğinde internette arama yapar.
* `max_results=2`, dönen sonuç sayısını sınırlayarak cevabı sade tutar.
* `tools`, agente verilecek araç listesidir.

#### 4. Hafıza / Checkpointer
```python
memory = InMemorySaver()
```
* Bu yapı sohbet geçmişini RAM içinde tutar.
* Aynı oturum içinde konuşmayı devam ettirmek için kullanılır.
* Program kapanınca hafıza silinir, yani kalıcı değildir.

#### 5. Agent Oluşturma
```python
agent_executor = create_react_agent(llm, tools, checkpointer=memory)
```
* Bu satır model ile araçları birleştirip gerçek agent yapısını kurar.
* Agent, bazen doğrudan cevap verir, bazen önce arama aracını çalıştırır.
* `checkpointer=memory`, konuşma geçmişini `thread_id` ile eşleştirir.

#### 6. Oturum Ayarı
```python
config = {"configurable": {"thread_id": "abc123"}}
```
* `thread_id`, sohbet oturumunu tanımlar.
* Aynı `thread_id` kullanılırsa agent önceki konuşmayı devam ettirebilir.

#### 7. Döngü ve Streaming Çıktı
```python
if __name__ == "__main__":
    while True:
        user_input = input("> ")

        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        ):
            print(chunk, end="", flush=True)
            print("--")
```
* `while True`, programı kullanıcı durdurana kadar açık tutar.
* `HumanMessage(content=user_input)`, kullanıcı sorusunu LangChain mesaj formatına çevirir.
* `.stream()`, cevabı parça parça ekrana basar.
* Böylece terminalde canlı sohbet hissi oluşur.

### `main.py` Ne İşe Yarıyor?
Kısaca `main.py`, bu agent uygulamasının giriş noktasıdır. Şunları yapar:

1. `.env` içindeki değişkenleri yükler,
2. Groq modelini oluşturur,
3. Tavily web arama aracını ekler,
4. `InMemorySaver` ile geçici hafıza tutar,
5. ReAct agent’i kurar,
6. ve terminalde etkileşimli sohbet döngüsünü başlatır.

### Önemli Notlar
* Bu örnek, düz bir chatbot yerine araç kullanan bir agent yapısı gösterir.
* Agent gerektiğinde web araması yapabilir.
* Hafıza yalnızca uygulama çalıştığı sürece geçerlidir.
* `thread_id`, konuşmaların birbirine karışmaması için önemlidir.
