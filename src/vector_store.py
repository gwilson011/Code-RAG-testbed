"""
Vector store implementation using ChromaDB.
"""

from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import Config
from .ingest import CodeChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages code embeddings in ChromaDB."""

    def __init__(
        self,
        collection_name: str = None,
        db_path: str = None,
        reset: bool = False
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            db_path: Path to ChromaDB storage
            reset: Whether to reset/clear the collection
        """
        self.collection_name = collection_name or Config.CHROMA_COLLECTION_NAME
        self.db_path = Path(db_path or Config.CHROMA_DB_PATH)

        # Create ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Reset collection: {self.collection_name}")
            except Exception:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Code embeddings for RAG"}
        )

        logger.info(f"Initialized vector store with collection: {self.collection_name}")

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        batch_size: int = 100
    ):
        """
        Add code chunks with their embeddings to the vector store.

        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors
            batch_size: Number of items to add per batch
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            # Prepare data for ChromaDB
            ids = [f"{chunk.filepath}:{chunk.chunk_id}" for chunk in batch_chunks]
            documents = [chunk.content for chunk in batch_chunks]
            metadatas = [chunk.metadata for chunk in batch_chunks]

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Added batch {i // batch_size + 1} ({len(batch_chunks)} chunks)")

        logger.info(f"Successfully added {len(chunks)} chunks to vector store")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar code chunks.

        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary with query results including documents, distances, and metadata
        """
        top_k = top_k or Config.TOP_K_RESULTS

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "db_path": str(self.db_path)
        }

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Code embeddings for RAG"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")
