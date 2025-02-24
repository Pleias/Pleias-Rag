from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC

# Todo: Import chunking strategy functions here, we wont use objects anymore
from .database_chunking_strategies.BasicChunker import BasicChunker

@dataclass
class Document:
    """A document to be stored in the database."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: str = field(init=False)

@dataclass
class ChunkedDocument:
    """A chunk of a document, created by splitting the original document."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_doc_id: str

class RagDatabase:
    """
    A database for storing and searching documents with text chunking capabilities.
    """

    def __init__(
        self, 
        search_type: str = "bm25",
    ) -> None:
        """
        Initialize the database.
        
        Args:
            search_type: Type of search to use. Options:
                - "bm25" (default): BM25 text search
                - "vector": Vector-based search
        """
        if search_type not in ["bm25", "vector"]:
            raise ValueError('search_type must be either "bm25" or "vector"')

        self.search_type = search_type
        self.documents: List[Document] = []
        self.chunked_documents: List[ChunkedDocument] = []
        self._doc_counter: int = 0



    def add_documents(
        self, 
        documents: List[Union[Document, Tuple[str, Dict[str, Any]], str]]
    ) -> None:
        """
        Add multiple documents to the database and chunk them.
        
        Args:
            documents: List of either:
                - Document objects
                - Tuples of (text, metadata)
                - Strings (just the text)
        
        Example:
            >>> # Different ways to add documents
            >>> db.add_documents([
            ...     "First document",  # Just text
            ...     ("Second document", {"author": "John"}),  # With metadata
            ...     Document(text="Third document", metadata={"source": "web"})
            ... ])
        """
        docs_to_add = []
        for doc in documents:
            if isinstance(doc, Document):
                doc._id = f"doc_{self._doc_counter}"
                self._doc_counter += 1
                docs_to_add.append(doc)
            elif isinstance(doc, tuple) and len(doc) == 2:
                text, metadata = doc
                new_doc = Document(text=text, metadata=metadata)
                new_doc._id = f"doc_{self._doc_counter}"
                self._doc_counter += 1
                docs_to_add.append(new_doc)
            elif isinstance(doc, str):
                new_doc = Document(text=doc)
                new_doc._id = f"doc_{self._doc_counter}"
                self._doc_counter += 1
                docs_to_add.append(new_doc)
            else:
                raise ValueError(
                    "Each document must be either a Document object, "
                    "a tuple of (text, metadata), or just text"
                )
        
        # Add all documents and create their chunks
        for doc in docs_to_add:
            self.documents.append(doc)
            chunks = self.chunk_text(doc.text)
            chunked_docs = [
                ChunkedDocument(
                    chunk_id=f"{doc._id}_chunk_{i}",
                    text=chunk,
                    metadata=doc.metadata.copy(),
                    parent_doc_id=doc._id
                )
                for i, chunk in enumerate(chunks)
            ]
            self.chunked_documents.extend(chunked_docs)

    def __str__(self) -> str:
        return (
            f"RagDatabase(search_type={self.search_type}, "
            f"chunking_strategy={self.chunking_strategy}, "
            f"num_documents={len(self.documents)}, "
            f"num_chunks={len(self.chunked_documents)})"
        )
# Quick manual test:

db = RagDatabase()
docs = [
    "First document",
    ("Second document", {"author": "John"}),
    Document(text="Third document", metadata={"source": "web"})
]
db.add_documents(docs)

print("Documents:")
print(db.documents)

print("\nChunked Documents:")
print(db.chunked_documents)

print("\nFirst Chunk Text:")
print(db.chunked_documents[0].text)

print("\nSecond Chunk Metadata:")
print(db.chunked_documents[1].metadata)

print("\nThird Chunk Parent Document ID:")
print(db.chunked_documents[2].parent_doc_id)

print("\nThird Chunk ID:")
print(db.chunked_documents[2].chunk_id)

print("\nDatabase Summary:")
print(db)