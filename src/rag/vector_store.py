import logging
import os
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.errors import InternalError
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages interactions with the ChromaDB vector database.
    """
    def __init__(self, persist_directory: str, embedding_service: EmbeddingService):
        """
        Initialize the VectorStore.
        
        Args:
            persist_directory: Directory to store the database.
            embedding_service: Service to generate embeddings.
        """
        self.persist_directory = os.path.abspath(persist_directory)
        self.embedding_service = embedding_service

        os.makedirs(self.persist_directory, exist_ok=True)

        test_file = os.path.join(self.persist_directory, ".write_test")
        try:
            with open(test_file, "w", encoding="utf-8") as fh:
                fh.write("ok")
        except OSError as exc:
            raise RuntimeError(
                f"Vector store directory is not writable: {self.persist_directory}"
            ) from exc
        finally:
            if os.path.exists(test_file):
                try:
                    os.remove(test_file)
                except OSError:
                    pass

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except InternalError as exc:
            message = str(exc).lower()
            if "readonly" in message:
                raise RuntimeError(
                    "ChromaDB could not write to the vector store directory "
                    f"'{self.persist_directory}'. "
                    "Set STOCK_TRACKER_RAG_DIR to a writable location."
                ) from exc
            raise
        
    def get_collection(self, name: str = "stock_knowledge"):
        """
        Get or create a collection.
        """
        return self.client.get_or_create_collection(name=name)

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], collection_name: str = "stock_knowledge"):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of text content.
            metadatas: List of metadata dictionaries.
            ids: List of unique IDs for the documents.
            collection_name: Name of the collection.
        """
        if not documents:
            return
            
        collection = self.get_collection(collection_name)
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings(documents)
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")

    def query_similar(
        self,
        query: str,
        n_results: int = 5,
        collection_name: str = "stock_knowledge",
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query: The query text.
            n_results: Number of results to return.
            collection_name: Name of the collection.
            
        Returns:
            Query results from ChromaDB.
        """
        collection = self.get_collection(collection_name)
        
        query_embedding = self.embedding_service.generate_embeddings([query])
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where
        )
        return results
