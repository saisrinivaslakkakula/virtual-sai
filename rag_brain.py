import time
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.prompts import QuestionAnswerPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

LOG_FILE = "virtual_sai_debug.log"

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} {msg}\n")

def load_rag_engine():
    start = time.time()
    log("🧠 Loading documents...")
    documents = SimpleDirectoryReader("data").load_data()

    log("📦 Loading local embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

    log("🤖 Loading LLM (Ollama)...")
    llm = Ollama(model="mistral", temperature=0.2, request_timeout=60.0)

    log("📚 Indexing documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )

    log(f"✅ Done. Brain loaded in {round(time.time() - start, 2)} seconds.")
    return index, llm, embed_model

def get_answer(engine, query, persona_mode, llm, embed_model):
    log(f"🧠 Query Received: {query}")
    log(f"Persona: {persona_mode}")

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
    retriever = VectorIndexRetriever(index=engine)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=prompt,
        llm=llm,
        embed_model=embed_model
    )

    log("🧠 Executing query via query engine...")
    response = query_engine.query(query)
    log(f"✅ Response generated.")
    return response.response
