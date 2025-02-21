from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
from database_chunking_strategies import BasicChunker, BasicChunkingParams
from dataclasses import dataclass, field  # field was not imported
@dataclass
class ChunkingParams(ABC):
    """Abstract base class for chunking parameters."""
    pass


class SearchType(Enum):
    VECTOR = "vector"
    BM25 = "bm25"

@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    _id: str = field(init=False)  # Internal ID, not set by user

@dataclass
class ChunkedDocument:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_doc_id: str

class ChunkingStrategy(Enum):
    """
    Enumeration of available text chunking strategies.
    
    Available strategies:
        BASIC: Splits text based on word count while trying to maintain sentence integrity
    
    Note: Chunking parameters should also be provided when using a strategy.
    """
    BASIC = "basic_chunking"
    NONE = "no_chunking"

class RagDatabase:
    def __init__(self, 
                 search_type: SearchType = SearchType.BM25,
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.BASIC,
                 chunking_params: Optional[ChunkingParams] = None) -> None:
        self.search_type = search_type
        self.documents: List[Document] = []
        self.chunked_documents: List[ChunkedDocument] = []
        self.chunking_strategy = chunking_strategy
        self.chunking_params = chunking_params or self._get_default_params()
        self._doc_counter: int = 0  # Keep track of document count for IDs

        # We initialize a chunker based on the selected strategy
        if self.chunking_strategy == ChunkingStrategy.BASIC:
            self.chunker = BasicChunker(max_segment=self.chunking_params.max_segment)
        # if self.chunking_strategy == ChunkingStrategy.NONE:
        #     self.chunker = 
        else:
            raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")
    
    def _get_default_params(self) -> ChunkingParams:
        """
        Get default chunking parameters for the selected strategy.
        
        Returns:
            ChunkingParams: Default chunking parameters
        """
        if self.chunking_strategy == ChunkingStrategy.BASIC:
            return BasicChunkingParams()
        else:
            raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using the selected chunking strategy.
        
        Args:
            text (str): The text to be chunked

        Returns:
            List[str]: List of text chunks
        """ 
        return self.chunker.chunk_text(text)


    def add_documents(
        self, 
        documents: List[Union[Document, Tuple[str, Dict[str, Any]], str]]  # Now can take just text or (text, metadata)
    ) -> None:
        """
        Add multiple documents to the database and chunk them according to the current strategy.
        
        Args:
            documents: List of either:
                - Document objects
                - Tuples of (text, metadata) where metadata is optional
                - Strings (just the text)
        
        Example:
            >>> # Adding with just text
            >>> docs = [
            ...     "First document",
            ...     ("Second document", {"author": "John"}),  # Optional metadata
            ...     Document(text="Third document", metadata={"source": "web"})
            ... ]
            >>> db.add_documents(docs)
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
        

            
            

