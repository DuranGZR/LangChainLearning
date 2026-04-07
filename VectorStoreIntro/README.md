# Vector Store and RAG Introduction (LangChain & Chroma)

*Read this document in [English](#english-version) or [Turkish](#turkish-version).*

---

<a id="english-version"></a>
## 🇬🇧 English Version

### Project Overview
This project serves as an introduction to the **Retrieval-Augmented Generation (RAG)** architecture using **LCEL (LangChain Expression Language)**. It demonstrates how to initialize a local vector database, generate semantic document embeddings using open-source HuggingFace models, and seamlessly connect a local retriever to a cloud-based LLM. Running locally means no API keys are needed for the text-embedding phase, allowing for cost-free document indexing.

### Code-by-Code Explanation

#### 1. Imports and Environment Setup
```python
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
```
*   **dotenv**: Loads the `.env` file containing the `GROQ_API_KEY` for our LLM.
*   **LangChain Modules**: Imports the core classes for Document structures, Vector Stores (`Chroma`), external Embeddings (`HuggingFaceEmbeddings`), Models (`ChatGroq`), and Prompt templates.
*   **LCEL Runnables**: `RunnablePassthrough` is used to carry the user's string query forward through our pipeline.

#### 2. Document Creation
```python
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    # ...other documents omitted for brevity
]
```
*   **`Document` Object**: In reality, you would read these from a PDF or text file using a LangChain Document Loader. Here, we manually create them to map raw strings (`page_content`) alongside tracking extra info (`metadata`).

#### 3. Vectorization and Database Initialization
```python
vector_store = Chroma.from_documents(
    documents = documents,
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
```
*   **`HuggingFaceEmbeddings`**: Uses the `all-MiniLM-L6-v2` model from `sentence-transformers`. This locally runs on the CPU/GPU to map words mathematically without relying on a paid service like OpenAI.
*   **`Chroma.from_documents`**: Automatically ingests our text, runs them through the embedding model mapped above, and saves them in the active Chroma vector database.

#### 4. The Retriever Concept
```python
reteiever = vector_store.as_retriever(search_kwargs={"k": 1})
```
*   **`.as_retriever()`**: Converts the vector database into an LCEL-compatible tool. 
*   **`k: 1`**: Instructs the database to return only the top 1 single most semantically similar document matching the user's question.

#### 5. Prompt Engineering and LLM
```python
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
```
*   **`temperature=0.2`**: Low temperature is heavily preferred in RAG because we want factual answers rooted *strictly* in the provided text context, not creative hallucinations.
*   **Prompt Variables `{question}` and `{context}`**: These placeholders are critical; they tell LCEL precisely where to inject the retrieved vectors and user input right before sending them to the LLM.

#### 6. LCEL Pipeline Execution
```python
chain = {"context": reteiever , "question": RunnablePassthrough()} | promt | llm

if __name__ == "__main__":
    response = chain.invoke("tell me about dogs")
    print(response.content)
```
*   **`RunnablePassthrough()`**: Acts as a sponge that grabs the initial string (`"tell me about dogs"`) and funnels it identically to two places: the Retriever (to find identical context) and the Prompt (to be attached as the question).
*   **`|` (LCEL)**: The variables are evaluated into a dictionary, shipped to the prompt formatter, passed as message blocks into Groq, and the finalized model text is stored inside the `response.content` property.

### 🧠 Key Takeaways
* You learned how to pair **HuggingFace** local embedding operations with a **Chroma** datastore without needing external API logic.
* You discovered how to configure an LLM model dynamically inside an LCEL map `{"context": retriever, "question": RunnablePassthrough()}` so text retrieval occurs autonomously before the LLM takes a turn.

---

<a id="turkish-version"></a>
## 🇹🇷 Türkçe Versiyon

### Proje Özeti
Bu proje, **LCEL (LangChain Expression Language)** kullanılarak **Retrieval-Augmented Generation (RAG)** (Geri Getirmeyle Zenginleştirilmiş Üretim) mimarisine giriş yapmayı amaçlar. Açık kaynaklı HuggingFace modelleri kullanılarak kelimelerin nasıl yerel olarak anlamsal (semantic) bağlamda gömülüp (embedding) veri tabanında saklandığını gösterir. Vektörleştirme işlemleri yerel (local) çalıştığı için, OpenAI gibi ücretli servislere bağımlılık yaratmadan maliyetsiz bir doküman indeksleme ortamı deneyimi sunar.

### Kod Analizi: Neyin Neden Yapıldığı

#### 1. Kütüphanelerin Yüklenmesi
```python
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
```
*   **dotenv**: Dil modelimiz (`ChatGroq`) için gerekli olan `.env` dosyasındaki `GROQ_API_KEY`'i çeker.
*   **LangChain Modülleri**: Doküman nesneleri oluşturma, Vektör Veritabanı (`Chroma`), Gömme algoritmaları (`HuggingFaceEmbeddings`), Yapay zeka modeli (`ChatGroq`) ve akış bileşenlerini (Runnable) içe aktarıyoruz.
*   **LCEL Runnables**: `RunnablePassthrough`, kullanıcının girdiği yalın metni boru hattında bozmadan ilerletmek için kullanılır.

#### 2. Doküman (Dataset) Yapılandırması
```python
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    # ...diğer belgeler
]
```
*   **`Document` Nesnesi**: Gerçek senaryolarda bu kısmı PDF Loader veya TXT Loader araçlarıyla devasa dosyalardan çekeriz. Burada konuyu kavramak adına her cümlenin kendi içeriğini (`page_content`) ve nereden geldiğini belirten bilgisini (`metadata`) manuel tanımladık.

#### 3. Vektörleştirme ve Veritabanı (Embedding)
```python
vector_store = Chroma.from_documents(
    documents = documents,
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
```
*   **`HuggingFaceEmbeddings`**: `sentence-transformers` paketi üzerinden `all-MiniLM-L6-v2` algoritmasını çalıştırır. Cihazınızın işlemcisi kullanılarak sözcükler hiçbir API'ye gitmeden matematiksel vektör sayılara çevrilir.
*   **`Chroma.from_documents`**: Yukarıda ayırdığımız saf doküman metinlerini tek tek içeriye alır, vektör algoritmasından geçirir ve yeni bir geçici Chroma veritabanı belleğine kilitler.

#### 4. Retriever (Geri Getirici) Mantığı
```python
reteiever = vector_store.as_retriever(search_kwargs={"k": 1})
```
*   **`.as_retriever()`**: Statik halinde duran veritabanını, LCEL boru hattına entegre edilebilen, soru-cevap anında aktif arama sağlayabilen dinamik bir modüle dönüştürür.
*   **`k: 1`**: Sorulan soruya anlamsal açıdan *en yakın sadece 1 dokümanı/paragrafı* çekmesini belirtiriz.

#### 5. Prompt Mühendisliği ve Dil Modeli
```python
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
```
*   **`temperature=0.2`**: RAG projelerinde modelin yaratıcılığı veya uydurma (hallucination) ihtimali istenmez. Sorulan sorunun cevabı *SADECE* ona verdiğimiz bağlam içerisinde (context) bulunsun diye bu ayar iyice düşük tutulur.
*   **Prompt Değişkenleri `{question}` ve `{context}`**: Bu kıvrımlı parantezler hayati önem taşır; LCEL birazdan bu boşlukların içerisine arama sonuçlarımızı (context) ve kullanıcının sorduğu soruyu (question) adeta matbaada basar gibi oturtup modeli öyle besleyecektir.

#### 6. LCEL Zincirinin Aktif Edilmesi
```python
chain = {"context": reteiever , "question": RunnablePassthrough()} | promt | llm

if __name__ == "__main__":
    response = chain.invoke("tell me about dogs")
    print(response.content)
```
*   **`RunnablePassthrough()`**: Süngersi bir yapıdır. Birinci adımdaki sorumuz olan `"tell me about dogs"` string'ini bir kalıp gibi yutar ve tek bir cisme eşit bir şekilde iki farklı yöne iletir: Birinci yön Retriever modülünenin içi (ilgili dökümanı bul k=1), İkinci yön Question modülünün içi (Prompt'un içine soruyu oturt).
*   **`|` (LCEL)**: Bu sembol sayesinde elde edilen tüm yeni argümanlar sağa kayar; prompt içerisinde buluşur, oradan Llama3.3 modeline sızar ve nihai bir cevap oluşturularak kodumuzdaki statik konsola (`print(response.content)`) basılır.

### 🧠 Öğrenim Çıktıları
* **Chroma** veritabanı ile **HuggingFace**'i birleştirerek yapay zekadaki en büyük masraf kalemi olan API bağımlılığından nasıl kurtulacağınızı (lokal çalıştırmayı) öğrendiniz.
* Modele sorulan sorudan saniyeler hemen önce arkaplanda anlamsal bir arama operasyonunu çalıştırmayı ve `{"context": retriever, "question": RunnablePassthrough()}` hilesini kullanarak bunu LCEL mimarisine profesyonelce sokmayı öğrendiniz.
