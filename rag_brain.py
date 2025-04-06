from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core import ServiceContext
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.prompts import QuestionAnswerPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time

def load_rag_engine():
    start = time.time()
    print("🧠 Loading documents...")
    documents = SimpleDirectoryReader("data").load_data()

    print("📦 Loading local embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    print("🤖 Loading LLM...")
    llm = Ollama(model="phi", temperature=0.2, request_timeout=60.0)
  # MS2 swap later
    print("📚 Indexing documents...")
    index = VectorStoreIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model
)

    print(f"✅ Done. Brain loaded in {round(time.time() - start, 2)} seconds.")
    return index, llm, embed_model

def get_answer(engine, query, persona_mode, llm, embed_model):
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
    llm=llm,
    embed_model=embed_model
)

    response = custom_query_engine.query(query)
    return response.response
