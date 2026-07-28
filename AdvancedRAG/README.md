# Advanced RAG Pipeline (Routing, Grading, Web Search, and Generation)

*Read this document in [English](#english-version) or [Turkish](#turkish-version).* 

---

<a id="english-version"></a>
## 🇬🇧 English Version

### Project Overview
This project demonstrates an **advanced Retrieval-Augmented Generation (RAG)** workflow built with **LangGraph**. It goes beyond a basic vector search pipeline by adding three important control layers:

1. a **question router** that decides whether the query should go to the vector store or the web,
2. a **document grader** that filters out irrelevant retrieved chunks,
3. a **generation grader** that checks whether the final answer is grounded in the retrieved context and actually answers the question.

The result is a self-correcting RAG system that can retrieve from a local Chroma index, fall back to web search when the retrieved context is weak, and retry generation when the answer is unsupported or incomplete.

### Repository Structure
* [main.py](main.py) is the entry point and runs a single example question through the graph.
* [ingestion.py](ingestion.py) downloads source pages, splits them into chunks, embeds them, and builds the Chroma retriever.
* [graph/graph.py](graph/graph.py) defines the LangGraph workflow and the conditional routing logic.
* [graph/state.py](graph/state.py) defines the typed graph state shared across nodes.
* [graph/node_constants.py](graph/node_constants.py) stores the node names used by the graph.
* [graph/chains/](graph/chains) contains the LLM chains for routing, retrieval grading, hallucination grading, answer grading, and generation.
* [graph/nodes/](graph/nodes) contains the executable node functions used by the workflow.

### End-to-End Flow
The pipeline works like this:

1. `main.py` loads environment variables and invokes the compiled graph with a question.
2. The graph routes the question through the router chain.
3. If the router chooses `vectorstore`, the retriever fetches relevant chunks from Chroma.
4. Each retrieved chunk is graded for relevance.
5. If the retrieved context looks weak, the workflow can switch to web search.
6. The final answer is generated from the available documents.
7. The generated answer is checked for hallucination and usefulness.
8. If the answer is unsupported or does not answer the question, the workflow can retry or fall back to web search again.

This is what makes the project more advanced than a standard RAG example: it does not assume retrieval is always enough, and it does not trust the model output blindly.

### Code-by-Code Explanation

#### 1. Entry Point
```python
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is prompt engineering?"}))
```
* `load_dotenv()` loads environment variables from `.env` so API keys are kept out of the code.
* `app` is the compiled LangGraph workflow imported from `graph/graph.py`.
* `app.invoke(...)` sends a single question into the pipeline and returns the final state.
* This file is currently a smoke-test style entry point rather than an interactive chat loop.

#### 2. Ingestion and Vector Store Setup
```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0)

splits = text_splitter.split_documents(docs_list)
```
* `WebBaseLoader` downloads the contents of three Lilian Weng articles.
* The nested list of documents is flattened into `docs_list` so the splitter can process them consistently.
* `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)` creates short chunks sized for retrieval.
* `chunk_size=250` keeps each chunk small and focused.
* `chunk_overlap=0` means chunks do not overlap, so the split is very strict and compact.

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents = splits,
    collection_name = "rag-chroma",
    embedding = embeddings,
    persist_directory = "./.chroma"
)

retriever = Chroma(
    collection_name = "rag-chroma",
    embedding_function = embeddings,
    persist_directory = "./.chroma"
).as_retriever()
```
* `HuggingFaceEmbeddings` turns text chunks into vectors with a local embedding model.
* `Chroma.from_documents(...)` stores the embedded chunks in a persistent local vector database.
* `persist_directory="./.chroma"` makes the vector store reusable across runs.
* `retriever` exposes the vector store as a queryable retrieval tool for the graph.

#### 3. Typed Graph State
```python
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
```
* `question` stores the user query.
* `documents` keeps the retrieved or web-enriched context.
* `generation` stores the latest model answer.
* `web_search` acts as a decision flag that tells the graph whether the retrieval stage should fall back to the web.

#### 4. Routing the Question
```python
class RouterQuery(BaseModel):
    source: Literal["vectorstore", "websearch"]
```
* The router is a structured LLM output that returns a strict destination.
* The router only chooses between `vectorstore` and `websearch`.
* The prompt tells the model to use the vector store for questions about agents, prompt engineering, and adversarial attacks.
* Everything else is sent to web search.

```python
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
```
* The first decision in the graph is made before retrieval starts.
* This keeps the workflow flexible and avoids forcing every question through the vector store.

#### 5. Retrieving Relevant Documents
```python
def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```
* The retriever fetches the most similar chunks for the current question.
* The retrieved chunks are added to the graph state as `documents`.

#### 6. Grading Retrieved Documents
```python
for d in documents:
    score = retrieval_grader.invoke({
        "question": question,
        "document": d.page_content
    })
```
* Each retrieved chunk is checked individually.
* Relevant chunks are kept.
* Irrelevant chunks are filtered out.
* If any chunk fails relevance grading, the `web_search` flag is turned on so the graph can recover with external search.

This stage matters because retrieval is not perfect. A RAG system can become more accurate by refusing weak context instead of passing everything directly into the generator.

#### 7. Generation
```python
generation = generation_chain.invoke({"context": documents, "question": question})
```
* The generation chain formats the context, injects it into a prompt, sends it to Groq, and parses the response into text.
* The prompt instructs the model to answer concisely and to say when it does not know.

```python
def format_docs(inputs):
	documents = inputs["context"]
	return {
		"context": "\n\n".join(document.page_content for document in documents),
		"question": inputs["question"],
	}
```
* This helper converts a list of `Document` objects into a single text block.
* The formatting step makes the context easier for the model to consume.

#### 8. Hallucination and Answer Grading
```python
score = hallucination_grader.invoke({"documents": documents, "generation": generation})
score = answer_grader.invoke({"question": question, "generation": generation})
```
* `hallucination_grader` checks whether the answer is grounded in the retrieved documents.
* `answer_grader` checks whether the answer actually resolves the user question.
* If the answer is grounded and useful, the graph ends.
* If the answer is grounded but not useful, or not grounded at all, the graph can loop back into web search or try generation again.

This is the main safety layer of the project. It prevents the graph from accepting a polished but unsupported answer.

#### 9. Web Search Fallback
```python
web_search_tool = TavilySearchResults(k=3)
```
* Tavily is used as the external search fallback.
* The search results are converted into a `Document` and appended to the current context.
* The fallback lets the graph recover when the local corpus does not contain enough information.

#### 10. Workflow Assembly
```python
workflow = StateGraph(GraphState)
```
* The graph defines four main nodes: `retrieve`, `grade_documents`, `generate`, and `websearch`.
* Routing happens at the entry point.
* Document grading decides whether to generate or search the web.
* Generation is then graded for hallucinations and answer quality.

```python
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
```
* This is the self-correction loop.
* Unsupported generations retry generation.
* Useful generations end the graph.
* Weak answers can trigger web search.

### What `main.py` Does
In short, `main.py`:

1. loads environment variables,
2. imports the compiled LangGraph application,
3. sends one question into the workflow,
4. prints the final graph result.

### Requirements and Setup Notes
* A valid `GROQ_API_KEY` is required.
* Tavily credentials are required for the web search fallback.
* `ingestion.py` fetches external URLs during import, so the first run depends on network access.
* The project also writes a `graph.png` file when the graph is compiled.

### Key Takeaways
* This project is not a plain RAG example; it is a routed and self-checking RAG system.
* The router decides the first path, the grader filters retrieval noise, and the final graders verify answer quality.
* The workflow can fall back to web search when the local vector store is not enough.
* The vector store is persistent through Chroma, so the ingestion output can be reused across runs.

---

<a id="turkish-version"></a>
## 🇹🇷 Türkçe Versiyon

### Proje Özeti
Bu proje, **LangGraph** ile kurulmuş gelişmiş bir **Retrieval-Augmented Generation (RAG)** akışını gösterir. Basit bir vektör arama örneğinin ötesine geçer ve üç önemli kontrol katmanı ekler:

1. sorunun **vektör veritabanına mı yoksa web aramasına mı** gideceğine karar veren bir **router**,
2. geri getirilen parçaların gerçekten alakalı olup olmadığını ayıklayan bir **document grader**,
3. üretilen cevabın hem bağlama bağlı olup olmadığını hem de soruyu gerçekten çözüp çözmediğini kontrol eden bir **generation grader**.

Sonuç olarak, gerektiğinde yerel Chroma indeksinden bilgi çeken, context zayıfsa web aramasına düşen ve üretilen cevap tatmin edici değilse kendini yeniden deneyen bir RAG sistemi ortaya çıkar.

### Klasör Yapısı
* [main.py](main.py) giriş noktasıdır ve grafiği tek bir örnek soru ile çalıştırır.
* [ingestion.py](ingestion.py) kaynak sayfaları indirir, parçalara böler, embed eder ve Chroma retriever oluşturur.
* [graph/graph.py](graph/graph.py) LangGraph akışını ve koşullu yönlendirme mantığını tanımlar.
* [graph/state.py](graph/state.py) düğümler arasında paylaşılan tipli state yapısını tanımlar.
* [graph/node_constants.py](graph/node_constants.py) grafikte kullanılan düğüm isimlerini tutar.
* [graph/chains/](graph/chains) routing, retrieval grading, hallucination grading, answer grading ve generation zincirlerini içerir.
* [graph/nodes/](graph/nodes) işleyen node fonksiyonlarını içerir.

### Uçtan Uca Akış
Bu pipeline şu şekilde çalışır:

1. `main.py` ortam değişkenlerini yükler ve bir soruyu grafiğe verir.
2. Soru önce router chain üzerinden yönlendirilir.
3. Router `vectorstore` seçerse retriever Chroma içinden ilgili parçaları çeker.
4. Çekilen her parça relevance açısından grade edilir.
5. Eğer gelen context zayıfsa workflow web search’e kayabilir.
6. Son cevap eldeki dökümanlardan üretilir.
7. Üretilen cevap hallucination ve usefulness açısından kontrol edilir.
8. Cevap desteklenmiyorsa veya soruyu çözmüyorsa workflow tekrar deneyebilir ya da yeniden web search’e dönebilir.

Bu yüzden proje klasik bir RAG örneğinden daha gelişmiştir: her şeyi doğrudan modele bırakmaz ve retrieval çıktısına körü körüne güvenmez.

### Kod Analizi: Neyin Neden Yapıldığı

#### 1. Giriş Noktası
```python
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("Hello Advanced RAG")
    print(app.invoke(input={"question": "what is prompt engineering?"}))
```
* `load_dotenv()`, `.env` içindeki değişkenleri yükler ve API anahtarlarını koddan ayrı tutar.
* `app`, `graph/graph.py` içinden gelen derlenmiş LangGraph uygulamasıdır.
* `app.invoke(...)`, tek bir soruyu pipeline’a gönderir ve son state’i döndürür.
* Bu dosya şu an interaktif sohbet yerine smoke-test tarzı bir giriş noktasıdır.

#### 2. Ingestion ve Vector Store Kurulumu
```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0)

splits = text_splitter.split_documents(docs_list)
```
* `WebBaseLoader`, Lilian Weng’in üç yazısının içeriğini indirir.
* İçe içe gelen document listesi `docs_list` ile düzleştirilir.
* `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)`, metni retrieval için uygun boyutta parçalara böler.
* `chunk_size=250`, küçük ve odaklı chunk’lar üretir.
* `chunk_overlap=0`, chunk’lar arasında örtüşme bırakmaz; yani kesim oldukça sıkıdır.

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents = splits,
    collection_name = "rag-chroma",
    embedding = embeddings,
    persist_directory = "./.chroma"
)

retriever = Chroma(
    collection_name = "rag-chroma",
    embedding_function = embeddings,
    persist_directory = "./.chroma"
).as_retriever()
```
* `HuggingFaceEmbeddings`, metin parçalarını local bir embedding modeli ile vektöre dönüştürür.
* `Chroma.from_documents(...)`, embed edilmiş chunk’ları yerel ve kalıcı bir vektör veritabanına yazar.
* `persist_directory="./.chroma"`, veritabanının sonraki çalışmalarda yeniden kullanılmasını sağlar.
* `retriever`, Chroma’yı grafiğin sorgulayabileceği bir retrieval aracına dönüştürür.

#### 3. Tipli Graph State
```python
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
```
* `question`, kullanıcı sorusunu tutar.
* `documents`, geri getirilen veya web’den zenginleştirilen context’i tutar.
* `generation`, modelin son cevabını saklar.
* `web_search`, retrieval aşamasının web’e düşmesi gerekip gerekmediğini belirten karar bayrağıdır.

#### 4. Soruyu Yönlendirme
```python
class RouterQuery(BaseModel):
    source: Literal["vectorstore", "websearch"]
```
* Router, sıkı biçimde yapılandırılmış bir LLM çıktısıdır.
* Router sadece `vectorstore` ve `websearch` arasında seçim yapar.
* Prompt, agent, prompt engineering ve adversarial attack konularında vector store kullanılmasını söyler.
* Diğer tüm sorular web search’e gider.

```python
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
```
* Grafikte ilk karar retrieval başlamadan önce verilir.
* Bu sayede her soruyu zorla vector store’a sokmak gerekmez.

#### 5. İlgili Belgeleri Geri Getirme
```python
def retrieve(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
```
* Retriever, soruya en benzeyen chunk’ları çeker.
* Çekilen chunk’lar `documents` olarak state’e eklenir.

#### 6. Belgeleri Grade Etme
```python
for d in documents:
    score = retrieval_grader.invoke({
        "question": question,
        "document": d.page_content
    })
```
* Her retrieved chunk tek tek kontrol edilir.
* Alakalı chunk’lar tutulur.
* Alakasız chunk’lar elenir.
* Eğer en az bir chunk kötü görünüyorsa `web_search` bayrağı açılır ve graph dış kaynağa dönebilir.

Bu aşama önemlidir çünkü retrieval kusursuz değildir. Zayıf context’i direkt generator’a vermek yerine ellemek, RAG kalitesini ciddi şekilde artırır.

#### 7. Cevap Üretimi
```python
generation = generation_chain.invoke({"context": documents, "question": question})
```
* Generation chain context’i formatlar, prompt’a yerleştirir, Groq modeline yollar ve cevabı metne çevirir.
* Prompt, cevabın kısa olmasını ve bilinmeyen durumda uydurmamasını ister.

```python
def format_docs(inputs):
	documents = inputs["context"]
	return {
		"context": "\n\n".join(document.page_content for document in documents),
		"question": inputs["question"],
	}
```
* Bu yardımcı fonksiyon `Document` listesini tek bir metin bloğuna dönüştürür.
* Böylece model context’i daha rahat işler.

#### 8. Hallucination ve Answer Grading
```python
score = hallucination_grader.invoke({"documents": documents, "generation": generation})
score = answer_grader.invoke({"question": question, "generation": generation})
```
* `hallucination_grader`, cevabın retrieved documents ile desteklenip desteklenmediğini kontrol eder.
* `answer_grader`, cevabın soruyu gerçekten çözüp çözmediğini kontrol eder.
* Cevap hem grounded hem useful ise graph biter.
* Cevap yetersizse graph yeniden generation deneyebilir ya da web search’e dönebilir.

Bu proje için en kritik güvenlik katmanı budur. Güzel yazılmış ama dayanağı olmayan bir cevabın kabul edilmesini engeller.

#### 9. Web Search Fallback
```python
web_search_tool = TavilySearchResults(k=3)
```
* Tavily, dış kaynak fallback’i olarak kullanılır.
* Arama sonuçları bir `Document` haline getirilip mevcut context’e eklenir.
* Yerel corpus yeterli değilse sistem bu şekilde toparlanır.

#### 10. Workflow Kurulumu
```python
workflow = StateGraph(GraphState)
```
* Grafikte dört ana node vardır: `retrieve`, `grade_documents`, `generate` ve `websearch`.
* İlk karar entry point’te verilir.
* Document grading, generate ile web search arasında seçim yapar.
* Generation sonrasında hallucination ve answer grading ile kalite kontrolü yapılır.

```python
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
```
* Bu kısım self-correction döngüsüdür.
* Desteklenmeyen generation tekrar denenir.
* Faydalı generation graph’i bitirir.
* Zayıf cevaplar web search’e yönlenebilir.

### `main.py` Ne İşe Yarıyor?
Kısaca `main.py` şunları yapar:

1. ortam değişkenlerini yükler,
2. derlenmiş LangGraph uygulamasını içe aktarır,
3. tek bir soruyu workflow’a gönderir,
4. final graph sonucunu ekrana basar.

### Kurulum ve Dikkat Edilecek Noktalar
* Geçerli bir `GROQ_API_KEY` gerekir.
* Web search fallback’i için Tavily kimlik bilgileri gerekir.
* `ingestion.py`, import sırasında harici URL’leri çektiği için ilk çalıştırma internet bağlantısına bağlıdır.
* Graph derlenirken ayrıca bir `graph.png` dosyası oluşturulur.

### Önemli Notlar
* Bu proje düz bir RAG örneği değil, yönlendirmeli ve kendini kontrol eden bir RAG sistemidir.
* Router ilk yolu seçer, grader retrieval gürültüsünü temizler, final grader ise cevap kalitesini doğrular.
* Local vector store yetmezse sistem web search’e fallback yapabilir.
* Chroma persistent olduğu için ingestion çıktısı tekrar kullanılabilir.