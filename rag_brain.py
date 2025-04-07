import time
import os
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.prompts import QuestionAnswerPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 🪵 Log file path
LOG_FILE = "debug.log"

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} {msg}\n")
        f.flush()

def load_rag_engine():
    start = time.time()
    log("🧠 Loading documents...")

    # 🔧 Improved PDF handling
    reader = SimpleDirectoryReader(
        input_dir="data",
        file_extractor={".pdf": PDFReader()}
    )
    documents = reader.load_data()
    log(f"📄 Loaded {len(documents)} documents.")

    # 🔍 Peek into documents
    for i, doc in enumerate(documents):
        snippet = doc.text[:500].replace("\n", " ").replace("\r", " ")
        log(f"📃 Doc {i+1} preview: {snippet}...")

    log("🔤 Loading embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    log("✅ Embedding model loaded.")

    log("🤖 Loading Mistral via Ollama...")
    llm = Ollama(model="mistral", temperature=0.2, request_timeout=60.0)
    log("✅ LLM initialized.")

    log("📚 Creating index...")
    index = VectorStoreIndex.from_documents(
        documents,
        llm=llm,
        embed_model=embed_model
    )
    log(f"✅ Brain ready in {round(time.time() - start, 2)} sec.")
    return index, llm, embed_model

def get_answer(engine, query, persona_mode, llm, embed_model):
    log(f"💬 Received query: {query}")
    log(f"🧠 Persona mode: {persona_mode}")

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

    # 🔍 Configurable retriever
    retriever = VectorIndexRetriever(index=engine, similarity_top_k=5)

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=prompt,
        llm=llm,
        embed_model=embed_model
    )

    log("🚀 Executing query...")
    response = query_engine.query(query)

    if not response.response.strip():
        log("⚠️ No meaningful response returned.")
        return "🤖 Sorry, I couldn't find anything relevant in the documents."

    log("✅ Response generated.")
    return response.response
