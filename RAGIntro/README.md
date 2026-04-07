# Web Document RAG Pipeline (Retrieval-Augmented Generation)

*Read this document in [English](#english-version) or [Turkish](#turkish-version).*

---

<a id="english-version"></a>
## 🇬🇧 English Version

### Project Overview
This project demonstrates a full, end-to-end **Retrieval-Augmented Generation (RAG)** pipeline. Instead of relying purely on an LLM's pre-trained knowledge, this system dynamically scrapes a real web page, parses its content, splits the text into manageable chunks, and embeds them into a local vector database. When a user asks a question, the system retrieves the most relevant pieces of information from the database and uses them as context to generate an accurate, hallucination-free answer.

### Code-by-Code Explanation

#### 1. Web Data Loading & Parsing
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/" ),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content", "post-title","post-header")
        )
    )
)
docs = loader.load()
```
*   **`WebBaseLoader`**: Connects to the provided URL to scrape the HTML content.
*   **`bs4.SoupStrainer`**: Web pages are full of noise (menus, footers, sidebars). We use BeautifulSoup capabilities to extract *only* the specific HTML classes (`post-content`, `post-title`) that contain the actual article data, resulting in a clean text document.

#### 2. Text Splitting (Chunking)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
splits = text_splitter.split_documents(docs)
```
*   **`RecursiveCharacterTextSplitter`**: Embedding models and LLMs have token limits. We cannot feed an entire 10,000-word article at once into an embedding function efficiently. This class splits the document into smaller `1000`-character chunks.
*   **`chunk_overlap=200`**: To ensure we don't accidentally cut a sentence or a crucial context in half, we allow a 200-character overlap between consecutive chunks.

#### 3. Vectorization and Database Storage
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vectostore = Chroma.from_documents(
    documents=splits, 
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
retriever = vectostore.as_retriever()
```
*   **`HuggingFaceEmbeddings`**: Uses the free, locally-running `all-MiniLM-L6-v2` model to convert our text chunks into mathematical vectors (numbers).
*   **`Chroma.from_documents`**: Ingests all the vectorized chunks into a local vector database, enabling ultra-fast semantic similarity searches. 
*   **`.as_retriever()`**: Converts the database into a queryable tool for LangChain.

#### 4. LangSmith Prompt Hub Integration
```python
from langsmith import Client

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
```
*   **`client.pull_prompt`**: Instead of writing a complex RAG prompt from scratch, we connect to the LangChain Hub (LangSmith) and seamlessly pull down `"rlm/rag-prompt"`. This is a community-tested, highly optimized prompt specifically designed for instructing an LLM to answer questions strictly using retrieved context.

#### 5. RAG Chain Orchestration (LCEL)
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
*   **`format_docs`**: A helper function that takes the list of matched documents from the database and glues their text together separated by double newlines.
*   **`retriever | format_docs`**: LCEL pipes the user's question into the database, fetches the raw documents, and immediately passes them into our formatting function.
*   **`rag_chain`**: The ultimate pipeline. It simultaneously gathers the formatted `context` and the raw `question`, funnels them into the pulled Hub Prompt, sends that prompt to the `ChatGroq` model, and finally parses the output into a clean string.

#### 6. Execution Loop & Streaming
```python
if __name__ == "__main__":
    for chunk in rag_chain.stream("What is maximum inner product search?"):
        print(chunk, end="", flush=True)
```
*   **`.stream()`**: As the model reads the scraped webpage and formulates an answer to our question, it streams the output token-by-token (word-by-word) straight to our console, delivering a fast, ChatGPT-like user experience.

---

<a id="turkish-version"></a>
## 🇹🇷 Türkçe Versiyon

### Proje Özeti
Bu proje, uçtan uca eksiksiz bir **Retrieval-Augmented Generation (RAG)** (Geri Getirmeyle Zenginleştirilmiş Üretim) boru hattının (pipeline) nasıl kurulduğunu gösterir. Sadece modelin önceden eğitilmiş eski bilgisine güvenmek yerine, bu sistem gerçek bir web sayfasını tarar (scrape), içeriğini ayrıştırır, küçük okunabilir parçalara böler ve yerel bir vektör veritabanına gömer. Kullanıcı bir soru sorduğunda, sistem veritabanından en alakalı parçaları bulur ve bunları kullanarak "halüsinasyon (uydurma) içermeyen", tamamen bağlama (o web sitesine) sadık bir cevap üretir.

### Kod Analizi: Neyin Neden Yapıldığı

#### 1. Web Verisi Yükleme ve Ayrıştırma (Scraping)
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/" ),
    bs_kwargs= dict(
        parse_only = bs4.SoupStrainer(
            class_ = ("post-content", "post-title","post-header")
        )
    )
)
docs = loader.load()
```
*   **`WebBaseLoader`**: Verdiğimiz URL'ye bağlanarak hedef sitenin HTML yapısını tamamen indirir.
*   **`bs4.SoupStrainer`**: Web siteleri menüler, alt bilgiler, reklamlar gibi gereksiz verilerle (noise) doludur. Sadece makalenin odak noktasını almak için BeautifulSoup (`bs4`) kütüphanesinin yeteneklerini kullanarak filtremeyi sadece `post-content` ve `post-title` (içerik ve başlık) sınıflarına odakladık. Bu sayede veritabanına temiz metinler göndeririz.

#### 2. Metin Parçalama (Text Splitter / Chunking)
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
splits = text_splitter.split_documents(docs)
```
*   **`RecursiveCharacterTextSplitter`**: Ne dil modellerine ne de Vektör modellerine 10.000 kelimelik bir web makalesini tek seferde veremeyiz (Token limiti sorunu). Bu sınıf, devasa boyuttaki metnimizi her biri maksimum `1000` karakterden oluşan ufak, hazmedilebilir bilgi bloklarına (chunk) ayırır.
*   **`chunk_overlap=200`**: Metni bıçak gibi rasgele kestigimizde, önemli bir cümlenin yarısının kesilip diğer bloka düşmesini engellemek için bloklar arasında `200` karakterlik bir örtüşme (kesişim/pay) payı bırakırız. Bağlam (context) asla zedelenmez.

#### 3. Vektörleştirme ve Veritabanı
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

vectostore = Chroma.from_documents(
    documents=splits, 
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
retriever = vectostore.as_retriever()
```
*   **`HuggingFaceEmbeddings`**: Bilgisayarımızın kendi işlemcisinde çalışan ücretsiz `all-MiniLM-L6-v2` modelini kullanarak ayrıştırılmış Türkçe/İngilizce kelime öbeklerimizi matematiksel sayılara (vektörlere) çeviririz.
*   **`Chroma.from_documents`**: Sayılara çevrilmiş bu bilgi bloklarının hepsini yerel `Chroma` veritabanına kaydeder, bu sayede saniyeler içinde anlamsal benzerlik araması yapılabilecek mimariyi oluşturur.
*   **`.as_retriever()`**: Vektör veritabanını, LangChain boru hatlarının doğrudan arama yapabileceği bir "silah/araç" (retriever) moduna sokar.

#### 4. LangSmith Prompt Hub (Hazır Şablonlar)
```python
from langsmith import Client

client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
```
*   **`client.pull_prompt`**: RAG projelerinde dil modeline "Aşağıdaki metne bakarak cevap ver, bilmiyorsan uydurma" demek için çok uzun promptlar yazılır. Biz bunu sıfırdan yazmak yerine LangChain geliştiricilerinin LangSmith (Hub) üzerinde yayınladığı yüksek kaliteli ve test edilmiş endüstri standardı olan `"rlm/rag-prompt"` şablonunu doğrudan projemize (pull) çekiyoruz.

#### 5. RAG Zincirinin Kurulması (LCEL Orkestrasyonu)
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context" : retriever | format_docs, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
*   **`format_docs`**: Veritabanı bize liste halinde parçalar gönderir. Bu yardımcı fonksiyon bu listeyi alır ve parçalar arasına boş satırlar koyarak LCEL'in (Prompt'un) anlayacağı tek ve devasa bir "paragraf (string)" çıktısına çevirir.
*   **`retriever | format_docs`**: Gelen soru önce veritabanındaki arayıcıya (retriever) vurur, bulunan sonuçlar ise hiç beklemeden üstteki yardımcı fonksiyona girerek formatlanır. Her iki işlem de `context` (bağlam) anahtarına aktarılır.
*   **`rag_chain`**: Sistem LCEL operatörü olan `|` ile birleştirilir: Hazırlanmış veritabanı yanıtları (context) ve kullanıcının ham sorusu (question) bir paket halinde internetten çektiğimiz LangSmith RAG Prompt'una iletilir. Prompt doldurularak modelimize (`llm`) yollanır ve en sonunda tertemiz metin tipine dönüştürülüp (`StrOutputParser`) ekrana yansıtılır.

#### 6. Akış Sistemli Sunum (Streaming)
```python
if __name__ == "__main__":
    for chunk in rag_chain.stream("What is maximum inner product search?"):
        print(chunk, end="", flush=True)
```
*   **`.stream()`**: `invoke()` komutunda model tamamen düşünüp, paragrafı bitirene kadar sistem sessizce kilitli kalır. `.stream()` sayesinde, yapay zekanın bulup yazdığı cevaplar -tıpkı ChatGPT ekranındaki gibi- saniyesinde harf harf komut satırına (konsola) aktarılır. Bu kullanıcının uygulamanın hızlı çalıştığını hissetmesini ve gecikmesiz bir deneyim yaşamasını sağlar.
