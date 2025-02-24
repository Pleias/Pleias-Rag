from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC

# Import the simple chunking function
from .text_chunking import chunk_text  # This function takes max_segment as parameter

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
        default_max_segment: int = 512
    ) -> None:
        """
        Initialize the database.
        
        Args:
            search_type: Type of search to use. Options:
                - "bm25" (default): BM25 text search
                - "vector": Vector-based search
            default_max_segment: Default maximum segment length for chunking
        """
        if search_type not in ["bm25", "vector"]:
            raise ValueError('search_type must be either "bm25" or "vector"')

        self.search_type = search_type
        self.default_max_segment = default_max_segment
        self.documents: List[Document] = []
        self.chunked_documents: List[ChunkedDocument] = []
        self._doc_counter: int = 0

    def add_documents(
        self, 
        documents: List[Union[Document, Tuple[str, Dict[str, Any]], str]],
        max_segment: Optional[int] = None
    ) -> None:
        """
        Add multiple documents to the database and chunk them using the chunking function.
        
        Args:
            documents: List of either:
                - Document objects
                - Tuples of (text, metadata)
                - Strings (just the text)
            max_segment: Maximum segment length for chunking. If None, uses default_max_segment.
        """
        chunk_size = max_segment or self.default_max_segment
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
            chunks = chunk_text(doc.text, max_segment=chunk_size)
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

    def add_pre_chunked_documents(
        self,
        chunks: List[Tuple[str, str]],  # List of (chunk_text, original_text) tuples
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add pre-chunked documents directly to both documents and chunked_documents lists.
        
        Args:
            chunks: List of tuples containing (chunk_text, original_text)
            metadata: Optional metadata to attach to all documents
        """
        if not metadata:
            metadata = {}
            
        # Group chunks by their original text
        chunks_by_original = {}
        for chunk_text, original_text in chunks:
            if original_text not in chunks_by_original:
                chunks_by_original[original_text] = []
            chunks_by_original[original_text].append(chunk_text)
            
        # Create documents and chunks
        for original_text, chunk_texts in chunks_by_original.items():
            # Create the parent document
            doc = Document(text=original_text, metadata=metadata.copy())
            doc._id = f"doc_{self._doc_counter}"
            self._doc_counter += 1
            self.documents.append(doc)
            
            # Create chunks
            chunked_docs = [
                ChunkedDocument(
                    chunk_id=f"{doc._id}_chunk_{i}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    parent_doc_id=doc._id
                )
                for i, chunk_text in enumerate(chunk_texts)
            ]
            self.chunked_documents.extend(chunked_docs)

    def __str__(self) -> str:
        return (
            f"RagDatabase(search_type={self.search_type}, "
            f"default_max_segment={self.default_max_segment}, "
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