# rag_json/config.py

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class RAGConfig:
    # Embedding Model
    EMBEDDING_MODEL = "BAAI/bge-m3"
    # Gemini API Key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") 
    
    GEMINI_MODEL = os.getenv("MODEL_NAME", "") 

    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") # Path to rag_json/data

    EMBEDDINGS_FILE = os.path.join(_DATA_DIR, "plant_embeddings_full.pkl")
    PLANT_INDEX_FILE = os.path.join(_DATA_DIR, "plant_index_full.pkl")
    BM25_CORPUS_FILE = os.path.join(_DATA_DIR, "bm25_tokenized_corpus_full.pkl") # New
    BM25_MODEL_FILE = os.path.join(_DATA_DIR, "bm25_model_full.pkl")            # New
    FAISS_INDEX_FILE: str = os.path.join(_DATA_DIR, "faiss_index_full.index")

    # RAG Retrieval Parameters
    TOP_K_RETRIEVAL = 5 # Number of documents to retrieve for context
    MIN_SIMILARITY_THRESHOLD = 0.6 # Min cosine similarity for dense results (adjust as needed)

    # RRF Constant (Reciprocal Rank Fusion)
    # A higher k value means more emphasis on documents ranked lower in the individual lists.
    # Common values are 60.
    RRF_K = 60

    # Path to your Type 1 Plant Data JSON file
    PLANT_METADATA_JSON_PATH = os.path.join(_DATA_DIR, "plant_data.json") 
    PLANT_GRAPH_PATH = os.path.join(_DATA_DIR, "plant_graph.json") 
    ORIGINAL_METADATA_FILE = os.path.join(_DATA_DIR, "merge_metadata.json") 
    MAX_HISTORY_LENGTH = 10 # Max history length for RAG queries
    
    # === Performance Optimization Settings ===
    EMBEDDING_BATCH_SIZE: int = 16  
    QUERY_CACHE_SIZE: int = 1000    
    MAX_CONTEXT_LENGTH: int = 4000  
    
    # === FAISS Optimization Settings ===
    FAISS_NPROBE: int = 10          
    USE_GPU_FAISS: bool = False     
    FAISS_OMP_THREADS: int = 4      
    
    # === Memory Optimization ===
    USE_MEMORY_MAPPING: bool = True  
    ENABLE_QUANTIZATION: bool = False 
    @classmethod
    def setup_faiss_performance(cls):
        """Configure FAISS for optimal performance"""
        import faiss
        
        # Set OpenMP threads
        faiss.omp_set_num_threads(cls.FAISS_OMP_THREADS)
        
        # GPU settings (if available)
        if cls.USE_GPU_FAISS and faiss.get_num_gpus() > 0:
            print(f"🚀 FAISS GPU acceleration enabled ({faiss.get_num_gpus()} GPUs)")
            return True
        else:
            print(f"💻 Using CPU FAISS with {cls.FAISS_OMP_THREADS} threads")
            return False