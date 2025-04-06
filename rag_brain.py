from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.query_engine import RetrieverQueryEngine

def load_rag_engine():
    documents = SimpleDirectoryReader("data").load_data()
    llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
    embed_model = OpenAIEmbedding()
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    return index

def get_answer(engine, query, persona_mode):
    if "First-person" in persona_mode:
        prompt_template_str = (
            "You are Sai Srinivas, a software engineer. Answer in first-person point of view. "
            "Be clear, confident, and detailed.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer (as Sai):"
        )
    else:
        prompt_template_str = (
            "You are a helpful assistant describing the career and background of Sai Srinivas. "
            "Answer in third-person point of view.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer:"
        )

    prompt = QuestionAnswerPrompt(prompt_template_str)

    # 🔁 Manual retriever for 0.9.48 compatibility
    retriever = VectorIndexRetriever(index=engine)

    custom_query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=prompt,
    )

    response = custom_query_engine.query(query)
    return response.response
