"""
RAG retrieval pipeline for code question answering.
"""

from typing import List, Dict, Optional, Any
import logging
from openai import OpenAI

from .config import Config
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class CodeRAG:
    """RAG pipeline for code retrieval and question answering."""

    def __init__(
        self,
        vector_store: VectorStore = None,
        embedding_generator: EmbeddingGenerator = None,
        llm_model: str = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
            llm_model: OpenAI model to use for generation
        """
        Config.validate()

        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.llm_model = llm_model or Config.OPENAI_MODEL
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant code chunks for a query.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of retrieved code chunks with metadata
        """
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(query)

        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )

        # Format results
        retrieved_chunks = []
        for i in range(len(results["documents"])):
            retrieved_chunks.append({
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
                "distance": results["distances"][i],
                "id": results["ids"][i]
            })

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
        return retrieved_chunks

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer using retrieved context.

        Args:
            query: User's question
            context_chunks: Retrieved code chunks
            system_prompt: Optional custom system prompt

        Returns:
            Generated answer
        """
        # Build context from retrieved chunks
        context = self._build_context(context_chunks)

        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions about code. "
                "Use the provided code snippets to answer the user's question accurately. "
                "If the code snippets don't contain enough information, say so. "
                "Always cite which file the information comes from."
            )

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1
            )
            answer = response.choices[0].message.content
            logger.info(f"Generated answer for query: {query[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def query(
        self,
        query: str,
        top_k: int = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        End-to-end RAG query: retrieve context and generate answer.

        Args:
            query: User's question
            top_k: Number of chunks to retrieve
            include_sources: Whether to include source chunks in response

        Returns:
            Dictionary with answer and optionally source chunks
        """
        # Retrieve relevant chunks
        chunks = self.retrieve(query, top_k=top_k)

        # Generate answer
        answer = self.generate_answer(query, chunks)

        result = {"answer": answer}
        if include_sources:
            result["sources"] = chunks

        return result

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            filepath = chunk["metadata"].get("filepath", "unknown")
            start_line = chunk["metadata"].get("start_line", "?")
            end_line = chunk["metadata"].get("end_line", "?")

            context_parts.append(
                f"--- Source {i}: {filepath} (lines {start_line}-{end_line}) ---\n"
                f"{chunk['content']}\n"
            )

        return "\n".join(context_parts)


def create_rag_pipeline(reset_db: bool = False) -> CodeRAG:
    """
    Convenience function to create a RAG pipeline with default settings.

    Args:
        reset_db: Whether to reset the vector database

    Returns:
        Configured CodeRAG instance
    """
    vector_store = VectorStore(reset=reset_db)
    embedding_generator = EmbeddingGenerator()
    return CodeRAG(
        vector_store=vector_store,
        embedding_generator=embedding_generator
    )
