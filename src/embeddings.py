"""
Embedding generation for code chunks using OpenAI.
"""

from typing import List
import logging
from openai import OpenAI

from .config import Config
from .ingest import CodeChunk

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for code chunks using OpenAI."""

    def __init__(self, model: str = None):
        """
        Initialize the embedding generator.

        Args:
            model: OpenAI embedding model to use. Defaults to config setting.
        """
        Config.validate()
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = model or Config.OPENAI_EMBEDDING_MODEL

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.info(
                f"Generating embeddings for batch {i // batch_size + 1} "
                f"({len(batch)} texts)"
            )

            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Error generating batch embeddings: {e}")
                raise

        return embeddings

    def embed_code_chunks(
        self, chunks: List[CodeChunk], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for code chunks.

        Args:
            chunks: List of CodeChunk objects
            batch_size: Number of chunks to process per batch

        Returns:
            List of embedding vectors
        """
        texts = [chunk.content for chunk in chunks]
        logger.info(f"Generating embeddings for {len(chunks)} code chunks")
        return self.generate_embeddings_batch(texts, batch_size=batch_size)
