# ───────────── 🔧 Final Patch for init_empty_weights ─────────────
import transformers.modeling_utils
import torch
from contextlib import contextmanager
import os
import logging
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("virtual_sai_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextmanager
def init_empty_weights():
    """Context manager that initializes weights with empty tensors."""
    def _init_empty_weights(module):
        # Only handle basic modules that don't require specific initialization
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight = torch.nn.Parameter(torch.empty_like(module.weight))
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = torch.nn.Parameter(torch.empty_like(module.bias))
        elif isinstance(module, torch.nn.Embedding):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight = torch.nn.Parameter(torch.empty_like(module.weight))
        elif isinstance(module, torch.nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight = torch.nn.Parameter(torch.empty_like(module.weight))
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = torch.nn.Parameter(torch.empty_like(module.bias))
    
    # Apply to all modules
    for module in torch.nn.Module.__subclasses__():
        if hasattr(module, 'apply'):
            try:
                # Skip modules that require specific initialization parameters
                if module.__name__ in ['Linear', 'Embedding', 'LayerNorm']:
                    # Create a temporary instance to apply the function
                    temp_instance = module()
                    temp_instance.apply(_init_empty_weights)
            except Exception as e:
                # Silently skip modules that can't be initialized
                pass
    yield

transformers.modeling_utils.init_empty_weights = init_empty_weights

import time
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.prompts import QuestionAnswerPrompt
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

LOG_FILE = "virtual_sai_debug.log"

def log(msg):
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} {msg}\n")
        f.flush()

def load_rag_engine():
    """Initialize and return the RAG engine components."""
    try:
        # Initialize LLM
        logger.info("Initializing LLM with mistral (quantized)...")
        llm = Ollama(
            model="mistral",
            request_timeout=300.0,
            temperature=0.7,
            context_window=4096,
            additional_kwargs={
                "num_predict": 512,
                "num_gpu": 1,
                "num_thread": 4,
                "mmap": True,
                "num_ctx": 2048,
                "f16_kv": True,
                "low_vram": True
            }
        )
        
        # Test LLM connection
        try:
            logger.info("Testing LLM connection...")
            test_response = llm.complete("Test connection")
            logger.info(f"LLM test response received")
        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            raise Exception("Failed to connect to LLM. Please ensure Ollama is running and mistral model is installed.")
        
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        try:
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
        
        # Configure settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        
        # Initialize vector store index
        if not os.path.exists("data"):
            os.makedirs("data")
            with open("data/placeholder.txt", "w") as f:
                f.write("Initial placeholder document.")
        
        logger.info("Loading documents...")
        try:
            from llama_index.readers.file import PDFReader, DocxReader
            
            # Initialize readers
            pdf_reader = PDFReader()
            docx_reader = DocxReader()
            
            documents = []
            data_dir = "data"
            
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                try:
                    if filename.lower().endswith('.pdf'):
                        docs = pdf_reader.load_data(file_path)
                    elif filename.lower().endswith('.docx'):
                        docs = docx_reader.load_data(file_path)
                    elif filename.lower().endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            docs = [Document(text=text)]
                    else:
                        logger.warning(f"Skipping unsupported file: {filename}")
                        continue
                    
                    documents.extend(docs)
                    logger.info(f"Successfully loaded {filename}")
                    
                    # Log a preview of the content
                    for doc in docs:
                        preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                        logger.info(f"Document preview: {preview}")
                        
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    continue
            
            if not documents:
                logger.warning("No documents found in data directory!")
                raise Exception("No documents found to process. Please upload some documents first.")
            
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            # Create the index with the documents
            index = VectorStoreIndex.from_documents(documents)
            
            # Configure retriever with higher similarity top k
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=5
            )
            
            # Configure response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True,
                streaming=False
            )
            
            # Configure query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)],
                response_synthesizer=response_synthesizer
            )
            
            logger.info("RAG engine initialized successfully")
            return query_engine, llm, embed_model
            
        except Exception as e:
            logger.error(f"Error initializing RAG engine: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in load_rag_engine: {e}")
        raise

def get_answer(engine, query, persona_mode, llm, embed_model):
    log(f"💬 Query received: {query}")
    log(f"👤 Persona: {persona_mode}")

    # Determine if query needs context
    context_free_prompt = """You are Sai Srinivas. Determine if this query requires your personal context/background information to answer properly.
    Only respond with 'YES' if you need context about your work, experience, or background to answer.
    Respond with 'NO' if this is a general query that any AI can answer without specific context (like greetings, general questions, etc).
    Query: {query}
    Need context? (YES/NO):"""
    
    try:
        needs_context = llm.complete(context_free_prompt.format(query=query)).strip().upper()
        log(f"🤔 Context needed: {needs_context}")
        
        if needs_context != "YES":
            # Use direct LLM response for queries that don't need context
            if "First-person" in persona_mode:
                direct_prompt = """You are Sai Srinivas, a software engineer. Answer in first-person point of view.
                Be clear, confident, and friendly in your response.
                Question: {query}
                Answer (as Sai):"""
            else:
                direct_prompt = """You are describing Sai Srinivas, a software engineer. Answer in third-person point of view.
                Be clear, confident, and friendly in your response.
                Question: {query}
                Answer:"""
            
            response = llm.complete(direct_prompt.format(query=query))
            log(f"✅ Direct response generated: {response}")
            return response

    except Exception as e:
        log(f"⚠️ Error in context detection: {e}, falling back to RAG")
        # Fall back to RAG engine if context detection fails
    
    # Continue with RAG engine for context-needed queries
    if "First-person" in persona_mode:
        prompt_template_str = (
            "You are Sai Srinivas, a software engineer. Answer in first-person point of view. "
            "Be clear, confident, and detailed. If you don't have enough context to answer, "
            "say so rather than making things up.\n"
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
            "Answer in third-person point of view. If you don't have enough context to answer, "
            "say so rather than making things up.\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Question: {query_str}\n"
            "Answer:"
        )

    # Update settings for this query
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Create the prompt
    qa_prompt = QuestionAnswerPrompt(prompt_template_str)
    
    # Create a response synthesizer with our custom prompt
    response_synthesizer = get_response_synthesizer(
        text_qa_template=qa_prompt,
        response_mode="tree_summarize",
        use_async=True
    )
    
    # Create a new query engine with the custom response synthesizer
    query_engine = RetrieverQueryEngine(
        retriever=engine.retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
    )
    
    log("🚀 Executing RAG query...")
    
    # Get nodes for debugging
    nodes = engine.retriever.retrieve(query)
    log(f"📚 Retrieved {len(nodes)} nodes")
    for i, node in enumerate(nodes):
        log(f"Node {i+1} score: {node.score}")
        log(f"Node {i+1} content: {node.node.text[:200]}...")
    
    try:
        response = query_engine.query(query)
        log(f"✅ RAG response generated: {response.response}")
        return response.response
    except Exception as e:
        log(f"❌ Error generating RAG response: {e}")
        # If RAG fails, try direct LLM as last resort
        try:
            if "First-person" in persona_mode:
                fallback_prompt = """You are Sai Srinivas. Answer this question in first person, 
                acknowledging that you don't have detailed context but will try to help:
                Question: {query}
                Answer:"""
            else:
                fallback_prompt = """You are describing Sai Srinivas. Answer this question in third person, 
                acknowledging that you don't have detailed context but will try to help:
                Question: {query}
                Answer:"""
            
            response = llm.complete(fallback_prompt.format(query=query))
            log(f"✅ Fallback response generated: {response}")
            return response
        except:
            return "I apologize, but I encountered an error while trying to generate a response. Please try again or rephrase your question."
