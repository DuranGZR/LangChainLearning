# Simple Chatbot with Memory (Stateful LLM)

*Read this document in [English](#english-version) or [Turkish](#turkish-version).*

---

<a id="english-version"></a>
## 🇬🇧 English Version

### Project Overview
Large Language Models (LLMs) are inherently **stateless**, meaning they do not retain any memory of past interactions. Each prompt sent to the API is processed in isolation. 
This project demonstrates how to overcome this limitation by building a **Stateful Chatbot** using LangChain. By implementing a session-based memory architecture, we give the model the ability to "remember" previous interactions within a specific session, allowing for continuous, contextual conversations.

### Code-by-Code Explanation

#### 1. Imports and Environment Setup
```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
```
*   **dotenv**: Loads API keys (like `GROQ_API_KEY`) from a local `.env` file into the system's environment variables.
*   **LangChain Imports**: We import the necessary modules to construct the prompt, define message types (`HumanMessage`), handle the in-memory store (`InMemoryChatMessageHistory`), and finally orchestrate the history wrapping (`RunnableWithMessageHistory`).

#### 2. Model Initialization
```python
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
```
*   Initializes the `llama-3.3-70b-versatile` model via Groq's high-speed inference engine.
*   **`temperature=0.2`**: We deliberately use a low temperature setting. This makes the model's responses more deterministic and logical, which is crucial for maintaining consistent context in a conversational memory chain without hallucinating.

#### 3. Memory Store Strategy
```python
store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```
*   **`store = {}`**: Since this is a lightweight educational concept, a global Python dictionary acts as a temporary, in-memory data store. In production, this would be Redis, PostgreSQL, etc.
*   **`get_session_history`**: A factory function that dynamically returns the chat history for a specific `session_id`. If the session doesn't exist, it instantiates a fresh `InMemoryChatMessageHistory()` object.

#### 4. Prompt Engineering & Context Injection
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps answer questions."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model
```
*   **`MessagesPlaceholder`**: This is a powerful LangChain feature. It acts as a dynamic insertion point. It tells LangChain exactly *where* to retrieve the historical conversation log and inject it into the prompt payload before appending the new user input.
*   **`chain`**: Connects the template directly to the model using LCEL (LangChain Expression Language).

#### 5. The Memory Wrapper (Orchestrator)
```python
config = { "configurable": { "session_id": "firstchat" } }

with_messages_history = RunnableWithMessageHistory(chain, get_session_history)
```
*   **`RunnableWithMessageHistory`**: This is the core wrapper. It seamlessly intercepts the invocation, calls `get_session_history` based on the `session_id` provided in the `config`, injects those messages into the `MessagesPlaceholder`, gets the model's response, and then appends that new response back into our dictionary `store`.

#### 6. Execution Loop & Streaming
```python
if __name__ == "__main__":
    while True:
        user_input = input(">")
        for r in with_messages_history.stream([HumanMessage(user_input)], config=config):
            print(r.content, end=" ")
```
*   **`.stream()`**: Instead of waiting for the model to generate the entirety of its response before displaying it (`.invoke()`), we utilize generator streams to yield chunks (tokens) in real-time. This vastly improves the perceived latency and overall user experience.

---

<a id="turkish-version"></a>
## 🇹🇷 Türkçe Versiyon

### Proje Özeti
Büyük Dil Modelleri (LLM'ler) varsayılan olarak **"Stateless" (Zaman ve durum bağımsız, hafızasız)** çalışırlar. Bu, modele gönderilen her sorunun geçmişten bağımsız olarak sıfırdan değerlendirildiği anlamına gelir.
Bu proje, LangChain kütüphanesinin imkanlarını kullanarak yapay zekaya nasıl oturum ("Session") tabanlı bir **Hafıza (Memory)** kazandırabileceğimizi gösterir. Modelin önceki mesajları hatırlamasını sağlayarak, bağlamdan kopmadan sürekli ve akıcı sohbet edebilen profesyonel bir Chatbot mimarisi kurulmuştur.

### Kod Analizi: Neyin Neden Yapıldığı

#### 1. Kütüphanelerin Yüklenmesi
```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()
```
*   **dotenv**: Yerel `.env` dosyasındaki API anahtarlarını (örn: `GROQ_API_KEY`) projenin çevresel değişkenlerine yükler, böylece şifreleri koda yazmak zorunda kalmayız.
*   **LangChain**: Prompt oluşturmak, hafıza modüllerini (`InMemoryChatMessageHistory`) ve hafızayı sisteme bağlayacak olan esas orkestratörü (`RunnableWithMessageHistory`) projeye dahil ediyoruz.

#### 2. Model Seçimi ve Yapılandırma
```python
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
```
*   Groq altyapısı ile çalışan yüksek kapasiteli `llama-3.3-70b-versatile` modelini tanımlıyoruz.
*   **`temperature=0.2`**: Hafızaya dayalı sürekli bir sohbette modelin mantıksal çerçeveden kopmaması ve halüsinasyon görmemesi (saçmalamaması) için sıcaklık oranı bilerek düşük tutulmuştur.

#### 3. Hafıza Yönetimi ve Depolama
```python
store = {}

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```
*   **`store = {}`**: Bu proje öğrenme amaçlı olduğu için sunucu maliyeti gerektirmeyen, geçici bir Python Sözlük yapısı (`store = {}`) kullanılmıştır. Gerçek dünya senaryolarında (Production), bu yapı Redis veya PostgreSQL gibi kalıcı sunucu veritabanlarına taşınmalıdır.
*   **`get_session_history`**: Sisteme gelen her yeni sohbette (benzersiz bir `session_id` ile), sözlükte o kullanıcıya ait kayıtlı bir hafıza var mı diye bakar. Yoksa, o oturuma özel temiz bir `InMemoryChatMessageHistory()` objesi yaratıp geri döndürür.

#### 4. Prompt Şablonlama ve Bağlam Aktarımı
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps answer questions."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model
```
*   **`MessagesPlaceholder`**: Sistemin kalbi bu değişkendir. LLM'e şunu söyleriz: "Dinamik olarak birazdan sana bir mesaj geçmişi yollayacağım, o geçmişi al ve tam bu satıra konumlandır". Böylece yeni soru yazıldığında bağlam kaybolmaz.
*   **`chain`**: Langchain'in LCEL yapısı kullanılarak "Şablon" ve "Model" birbirine boru hattı (pipeline) şeklinde bağlanır.

#### 5. Hafıza Orkestrasyonu (Wrapper)
```python
config = { "configurable": { "session_id": "firstchat" } }

with_messages_history = RunnableWithMessageHistory(chain, get_session_history)
```
*   **`RunnableWithMessageHistory`**: LangChain'in sunduğu en kritik sarmalayıcı modüldür. İşleyişi şöyledir: Kullanıcıdan gelen isteği yakalar, `config` içindeki `session_id`'ye bakar, o id ile üstteki `get_session_history` fonksiyonunu çalıştırıp geçmişi çeker. Eski geçmiş ile yeni soruyu `MessagesPlaceholder` noktasına yerleştirip modele gönderir. Son olarak modelin yeni cevabını tekrar sözlüğün (store) içerisine yedekler. Zinciri tam otomatik bir döngüye sokan komuttur.

#### 6. Gerçek Zamanlı Çıktı Akışı (Streaming)
```python
if __name__ == "__main__":
    while True:
        user_input = input(">")
        for r in with_messages_history.stream([HumanMessage(user_input)], config=config):
            print(r.content, end=" ")
```
*   **`.stream()`**: Klasik metotlarda (`.invoke()`), model bütün paragrafı yazmayı bitirene kadar kullanıcı bekletilir. `.stream()` jeneratörü kullanılarak, yapay zekanın ürettiği her kelime anlık (gerçek zamanlı) olarak terminale dökülmüştür. Bu yöntem, gecikmeyi hissettirmeyerek son kullanıcıya çok daha akıcı bir deneyim sunar.
