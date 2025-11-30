"""
Code ingestion and chunking utilities.
Handles loading code files and splitting them into manageable chunks for embedding.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CodeDocument:
    """Represents a code document with metadata."""

    def __init__(self, content: str, filepath: str, language: str = "python"):
        self.content = content
        self.filepath = filepath
        self.language = language
        self.metadata = {
            "filepath": filepath,
            "language": language,
            "size": len(content),
            "content_hash": self._hash_content(content),
        }

    def __repr__(self):
        return f"CodeDocument(filepath={self.filepath}, language={self.language})"

    @staticmethod
    def _hash_content(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class CodeChunk:
    """Represents a chunk of code with metadata."""

    def __init__(
        self,
        content: str,
        filepath: str,
        chunk_id: int,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ):
        self.content = content
        self.filepath = filepath
        self.chunk_id = chunk_id
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = {
            "filepath": filepath,
            "chunk_id": chunk_id,
            "start_line": start_line if start_line is not None else 0,
            "end_line": end_line if end_line is not None else 0,
            "content_hash": self._hash_content(content),
        }

    def __repr__(self):
        return f"CodeChunk(filepath={self.filepath}, chunk_id={self.chunk_id})"

    @staticmethod
    def _hash_content(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class CodeIngestor:
    """Handles ingestion and chunking of code files."""

    SUPPORTED_EXTENSIONS = {".py", ".js", ".java", ".cpp", ".c", ".go", ".rs", ".ts"}

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the code ingestor.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_file(self, filepath: Path) -> Optional[CodeDocument]:
        """Load a single code file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            language = self._detect_language(filepath)
            return CodeDocument(content, str(filepath), language)

        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return None

    def load_directory(
        self, directory: Path, recursive: bool = True
    ) -> List[CodeDocument]:
        """
        Load all code files from a directory.

        Args:
            directory: Path to directory
            recursive: Whether to search subdirectories

        Returns:
            List of CodeDocument objects
        """
        documents = []
        pattern = "**/*" if recursive else "*"

        for filepath in directory.glob(pattern):
            if filepath.is_file() and filepath.suffix in self.SUPPORTED_EXTENSIONS:
                doc = self.load_file(filepath)
                if doc:
                    documents.append(doc)

        logger.info(f"Loaded {len(documents)} code files from {directory}")
        return documents

    def chunk_document(self, document: CodeDocument) -> List[CodeChunk]:
        """
        Split a code document into chunks.

        Args:
            document: CodeDocument to chunk

        Returns:
            List of CodeChunk objects
        """
        chunks = []
        content = document.content
        lines = content.split("\n")

        # Simple line-based chunking
        current_chunk = []
        current_size = 0
        chunk_id = 0
        start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.chunk_size and current_chunk:
                # Create chunk from accumulated lines
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        filepath=document.filepath,
                        chunk_id=chunk_id,
                        start_line=start_line,
                        end_line=i - 1,
                    )
                )

                # Handle overlap
                overlap_lines = []
                overlap_size = 0
                for line in reversed(current_chunk):
                    if overlap_size + len(line) + 1 <= self.chunk_overlap:
                        overlap_lines.insert(0, line)
                        overlap_size += len(line) + 1
                    else:
                        break

                current_chunk = overlap_lines + [line]
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_id += 1
                start_line = i - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            chunks.append(
                CodeChunk(
                    content=chunk_content,
                    filepath=document.filepath,
                    chunk_id=chunk_id,
                    start_line=start_line,
                    end_line=len(lines) - 1,
                )
            )

        logger.debug(f"Split {document.filepath} into {len(chunks)} chunks")
        return chunks

    def _detect_language(self, filepath: Path) -> str:
        """Detect programming language from file extension."""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
        }
        return extension_map.get(filepath.suffix, "unknown")


def ingest_codebase(
    codebase_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[CodeChunk]:
    """
    Convenience function to ingest an entire codebase.

    Args:
        codebase_path: Path to codebase directory
        chunk_size: Size of chunks in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of all code chunks
    """
    ingestor = CodeIngestor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = ingestor.load_directory(Path(codebase_path))

    all_chunks = []
    for doc in documents:
        chunks = ingestor.chunk_document(doc)
        all_chunks.extend(chunks)

    logger.info(f"Ingested {len(all_chunks)} chunks from {len(documents)} files")
    return all_chunks
