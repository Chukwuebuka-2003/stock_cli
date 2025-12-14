import logging
import re
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles processing of documents, including cleaning and chunking.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean raw text by removing extra whitespace and special characters.
        """
        # Remove HTML tags if present (basic check)
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text(separator=" ")
            
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to split.
            chunk_size: Maximum characters per chunk.
            overlap: Number of characters to overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            
            # If we are not at the end of the text, try to find a sentence break
            if end < text_len:
                # Look for the last period/newline within the chunk to break cleanly
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + (chunk_size // 2):
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
            # Prevent infinite loop if overlap is too big or logic fails
            if start >= end:
                start = end
                
        return chunks

    @staticmethod
    def process_document(content: str, metadata: Dict[str, Any], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Process a single document: clean, chunk, and prepare for vector store.
        
        Args:
            content: Raw text content.
            metadata: Metadata associated with the document.
            chunk_size: Size of chunks.
            
        Returns:
            List of dictionaries containing 'text', 'metadata', and 'id'.
        """
        cleaned_text = DocumentProcessor.clean_text(content)
        chunks = DocumentProcessor.chunk_text(cleaned_text, chunk_size=chunk_size)
        
        processed_docs = []
        base_id = metadata.get("source", "doc")
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            
            processed_docs.append({
                "text": chunk,
                "metadata": chunk_metadata,
                "id": f"{base_id}_chunk_{i}"
            })
            
        return processed_docs
