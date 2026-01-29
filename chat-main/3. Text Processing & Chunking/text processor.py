"""
Text Processing and Chunking Module
Cleans, normalizes, and chunks text for embedding
"""
import re
from typing import List, Dict
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_PAGE


class TextProcessor:
    """Processes and chunks text for embeddings"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize text processor
        
        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk (url, title, etc.)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        if len(cleaned_text) < self.chunk_size:
            # If text is smaller than chunk size, return as single chunk
            return [{
                'text': cleaned_text,
                'metadata': metadata or {},
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(cleaned_text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < len(cleaned_text):
                # Look for sentence ending punctuation
                sentence_end = self._find_sentence_boundary(cleaned_text, start, end)
                if sentence_end != -1:
                    end = sentence_end
            
            # Extract chunk
            chunk_text = cleaned_text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'metadata': metadata or {},
                    'chunk_index': chunk_index,
                    'char_start': start,
                    'char_end': end
                })
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= 0:
                start = 1
            
            # Limit number of chunks per page
            if chunk_index >= MAX_CHUNKS_PER_PAGE:
                break
        
        # Add total chunks count to each chunk
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """
        Find the nearest sentence boundary before the end position
        
        Args:
            text: Full text
            start: Start position
            end: End position
            
        Returns:
            Position of sentence boundary, or -1 if not found
        """
        # Look backwards from end for sentence-ending punctuation
        search_text = text[start:end]
        
        # Find last occurrence of sentence-ending punctuation
        for i in range(len(search_text) - 1, -1, -1):
            if search_text[i] in '.!?':
                # Make sure it's followed by space or end of text
                if i + 1 >= len(search_text) or search_text[i + 1].isspace():
                    return start + i + 1
        
        return -1
    
    def process_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        Process multiple pages into chunks
        
        Args:
            pages: List of page dictionaries with content
            
        Returns:
            List of all chunks from all pages
        """
        all_chunks = []
        
        for page in pages:
            content = page.get('content', '')
            
            if not content:
                continue
            
            # Create metadata for chunks
            metadata = {
                'url': page.get('url', ''),
                'title': page.get('title', ''),
                'description': page.get('description', '')
            }
            
            # Create chunks for this page
            chunks = self.create_chunks(content, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunk_statistics(self, chunks: List[Dict]) -> Dict:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_characters': 0
            }
        
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_characters': sum(chunk_sizes)
        }


class SemanticChunker:
    """
    Advanced chunker that splits text based on semantic boundaries
    (Alternative to simple character-based chunking)
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE):
        self.chunk_size = chunk_size
    
    def chunk_by_paragraphs(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by paragraphs, combining small paragraphs
        
        Args:
            text: Text to chunk
            metadata: Metadata for chunks
            
        Returns:
            List of paragraph-based chunks
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size and we have content
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': metadata or {},
                    'chunk_index': chunk_index
                })
                current_chunk = para
                chunk_index += 1
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': metadata or {},
                'chunk_index': chunk_index
            })
        
        # Add total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        return chunks