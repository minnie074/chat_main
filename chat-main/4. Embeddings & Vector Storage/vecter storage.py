
"""
Embeddings and Vector Store Module
Generates embeddings and manages vector database storage
"""
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from config.settings import (
    EMBEDDING_MODEL, VECTOR_DB_DIR, COLLECTION_NAME,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD
)


class EmbeddingGenerator:
    """Generates embeddings from text using sentence transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        
        # Use sentence-transformers directly for better performance
        self.model = SentenceTransformer(model_name)
        
        # Get embedding dimension
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dimension}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class VectorStore:
    """Manages vector database using ChromaDB"""
    
    def __init__(self, collection_name: str = COLLECTION_NAME, persist: bool = True):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the collection
            persist: Whether to persist data to disk
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        if persist:
            self.client = chromadb.PersistentClient(
                path=str(VECTOR_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Get or create collection
        self.collection = None
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            print(f"Loaded existing collection: {self.collection_name}")
        except:
            # Create new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Website content embeddings"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, chunks: List[Dict]) -> bool:
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            return False
        
        try:
            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            print(f"Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Prepare documents, metadatas, and ids
            documents = texts
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                # Create metadata
                metadata = {
                    'url': chunk['metadata'].get('url', ''),
                    'title': chunk['metadata'].get('title', ''),
                    'chunk_index': chunk.get('chunk_index', i),
                    'total_chunks': chunk.get('total_chunks', len(chunks))
                }
                metadatas.append(metadata)
                
                # Create unique ID
                chunk_id = f"{hash(chunk['metadata'].get('url', ''))}_{i}"
                ids.append(chunk_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Successfully added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata and scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity = 1 - distance  # Convert distance to similarity
                    
                    # Only include results above threshold
                    if similarity >= SIMILARITY_THRESHOLD:
                        formatted_results.append({
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'similarity': similarity,
                            'id': results['ids'][0][i]
                        })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []
    
    def clear_collection(self):
        """Clear all documents from collection"""
        try:
            # Delete collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate collection
            self._initialize_collection()
            
            print(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                'name': self.collection_name,
                'total_documents': count,
                'embedding_dimension': self.embedding_generator.embedding_dimension,
                'model': self.embedding_generator.model_name
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def check_if_url_indexed(self, url: str) -> bool:
        """
        Check if a URL has already been indexed
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is indexed, False otherwise
        """
        try:
            results = self.collection.get(
                where={"url": url},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False


# Alternative: LangChain integration (if needed)
class LangChainVectorStore:
    """
    Vector store using LangChain's abstractions
    (Alternative implementation for compatibility)
    """
    
    def __init__(self):
        from langchain.vectorstores import Chroma
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR)
        )
    
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """Add texts to vector store"""
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        self.vectorstore.persist()
    
    def similarity_search(self, query: str, k: int = TOP_K_RESULTS):
        """Search for similar documents"""
        return self.vectorstore.similarity_search(query, k=k)