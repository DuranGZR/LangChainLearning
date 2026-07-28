from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)


def format_docs(inputs):
	documents = inputs["context"]
	return {
		"context": "\n\n".join(document.page_content for document in documents),
		"question": inputs["question"],
	}


prompt = ChatPromptTemplate.from_messages(
	[
		(
			"system",
			"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you do not know the answer, say that you do not know. Keep the answer concise.",
		),
		("human", "Question: {question}\n\nContext: {context}"),
	]
)

generation_chain = RunnableLambda(format_docs) | prompt | llm | StrOutputParser()