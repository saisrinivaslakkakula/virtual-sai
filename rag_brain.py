from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core import ServiceContext  # (Still here in case some modules rely)
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.prompts import QuestionAnswerPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import time
import sys
import traceback

def load_rag_engine():
    start = time.time()
    print("🧠 STEP 1: Starting brain load sequence...", flush=True)

    try:
        print("📂 STEP 2: Reading documents from `data/` directory...", flush=True)
        documents = SimpleDirectoryReader("data").load_data()
        print(f"✅ Loaded {len(documents)} documents.", flush=True)
    except Exception as e:
        print("❌ ERROR loading documents:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None, None, None

    try:
        print("🔤 STEP 3: Initializing HuggingFace embedding model: BAAI/bge-small-en", flush=True)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        print("✅ Embedding model initialized.", flush=True)
    except Exception as e:
        print("❌ ERROR loading embedding model:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None, None, None

    try:
        print("🧠 STEP 4: Loading Ollama LLM with model: phi", flush=True)
        llm = Ollama(model="mistral", temperature=0.2, request_timeout=60.0)
        print("✅ LLM initialized.", flush=True)
    except Exception as e:
        print("❌ ERROR loading LLM model:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None, None, None

    try:
        print("📈 STEP 5: Creating index from documents...", flush=True)
        index = VectorStoreIndex.from_documents(
            documents,
            llm=llm,
            embed_model=embed_model
        )
        print(f"✅ Index built successfully in {round(time.time() - start, 2)} seconds.", flush=True)
    except Exception as e:
        print("❌ ERROR building index:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        return None, None, None

    return index, llm, embed_model


def get_answer(engine, query, persona_mode, llm, embed_model):
    print(f"🧠 Running query: {query}", flush=True)

    if "First-person" in persona_mode:
        print("👤 Responding in FIRST-person mode", flush=True)
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
        print("📄 Responding in THIRD-person mode", flush=True)
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

    try:
        prompt = QuestionAnswerPrompt(prompt_template_str)

        retriever = VectorIndexRetriever(index=engine)
        print("🔍 Retriever initialized.", flush=True)

        custom_query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=prompt,
            llm=llm,
            embed_model=embed_model
        )
        print("🚀 Query engine ready. Executing query...", flush=True)

        response = custom_query_engine.query(query)
        print("✅ Query executed successfully.", flush=True)

        return response.response

    except Exception as e:
        print("❌ ERROR during answer generation:", e, flush=True)
        traceback.print_exc(file=sys.stdout)
        return "Sorry, I encountered an error while answering that question."
