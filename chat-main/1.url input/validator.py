"""
Configuration settings for the Website Chatbot
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"  # Can be changed to gpt-4, llama2, etc.
LLM_TEMPERATURE = 0.1  # Low temperature for factual responses
LLM_MAX_TOKENS = 500

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternative: "text-embedding-ada-002" for OpenAI embeddings

# Vector Database Configuration
VECTOR_DB = "chromadb"  # chromadb, faiss, pinecone
COLLECTION_NAME = "website_content"

# Text Processing Configuration
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks for context preservation
MAX_CHUNKS_PER_PAGE = 50  # Limit chunks per page to prevent overload

# Retrieval Configuration
TOP_K_RESULTS = 4  # Number of relevant chunks to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score

# Crawling Configuration
MAX_CRAWL_DEPTH = 2  # Maximum depth for crawling (0 = single page only)
MAX_PAGES = 10  # Maximum number of pages to crawl
REQUEST_TIMEOUT = 10  # Timeout for HTTP requests in seconds
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Memory Configuration
CONVERSATION_MEMORY_SIZE = 5  # Number of previous exchanges to remember

# API Keys (loaded from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Response Templates
NOT_FOUND_RESPONSE = "The answer is not available on the provided website."
ERROR_RESPONSE = "I encountered an error processing your request. Please try again."

# Content Extraction Settings
REMOVE_ELEMENTS = [
    'header', 'footer', 'nav', 'aside', 
    'script', 'style', 'iframe', 'noscript',
    'ads', 'advertisement', 'banner'
]

MIN_TEXT_LENGTH = 50  # Minimum length for text to be considered valid