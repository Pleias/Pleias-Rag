from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json

# Import the simple chunking function
from .database_chunking_strategies.chunk_text import chunk_text

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
    parent_doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class RagDatabase:
    """
    A database for storing and searching documents with text chunking capabilities.
    Includes LanceDB integration for persistent storage and vector search.
    """

    def __init__(
        self, 
        search_type: str = "bm25",
        default_max_segment: int = 300,
        db_path: str = "data/lancedb",
        embeddings_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """Initialize the database."""
        if search_type not in ["bm25", "vector"]:
            raise ValueError('search_type must be either "bm25" or "vector"')

        self.search_type = search_type
        self.default_max_segment = default_max_segment
        self.documents: List[Document] = []
        self.chunked_documents: List[ChunkedDocument] = []
        self._doc_counter: int = 0

        # Initialize LanceDB
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        
        # Initialize embeddings model if using vector search
        if search_type == "vector":
            self.embeddings_model = SentenceTransformer(embeddings_model)
            self.embedding_dim = self.embeddings_model.get_sentence_embedding_dimension()
        else:
            self.embeddings_model = None
            self.embedding_dim = 384  # Default for all-MiniLM-L6-v2

        # Create or get the chunks table
        if "chunks" not in self.db.table_names():
            # Create table with sample data
            sample_data = [{
                "chunk_id": "sample_chunk",
                "text": "sample text",
                "parent_doc_id": "sample_doc",
                "metadata_json": json.dumps({}),
                "vector": [0.0] * self.embedding_dim
            }]
            
            self.chunks_table = self.db.create_table(
                "chunks",
                data=sample_data,
                mode="overwrite"
            )
            
            # Clear the sample data
            self.chunks_table.delete("chunk_id = 'sample_chunk'")
        else:
            self.chunks_table = self.db.open_table("chunks")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the loaded model."""
        if self.embeddings_model is None:
            raise ValueError("Embeddings model not initialized. Use vector search mode.")
        return self.embeddings_model.encode(text)

    def add_and_chunk_documents(
        self, 
        documents: List[Union[Document, Tuple[str, Dict[str, Any]], str]],
        max_segment: Optional[int] = None
    ) -> None:
        """Add multiple documents to the database and chunk them."""
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
        lance_records = []
        for doc in docs_to_add:
            self.documents.append(doc)
            chunks = chunk_text(doc.text, max_segment=chunk_size)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc._id}_chunk_{i}"
                chunked_doc = ChunkedDocument(
                    chunk_id=chunk_id,
                    text=chunk,
                    parent_doc_id=doc._id,
                    metadata=doc.metadata.copy()
                )
                self.chunked_documents.append(chunked_doc)
                
                # Prepare record for LanceDB
                vector = (self._get_embedding(chunk) if self.search_type == "vector" 
                         else np.zeros(self.embedding_dim))
                
                lance_record = {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "parent_doc_id": doc._id,
                    "metadata_json": json.dumps(doc.metadata),
                    "vector": vector.tolist()
                }
                lance_records.append(lance_record)
        
        # Add to LanceDB
        if lance_records:
            self.chunks_table.add(lance_records)

    def add_pre_chunked_documents(
        self,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add pre-chunked documents directly."""
        if not metadata:
            metadata = {}
            
        lance_records = []
        for chunk in chunks:
            doc_id = f"doc_{self._doc_counter}"
            self._doc_counter += 1
            
            # Add as document
            doc = Document(text=chunk, metadata=metadata.copy())
            doc._id = doc_id
            self.documents.append(doc)
            
            # Add as chunked document
            chunked_doc = ChunkedDocument(
                chunk_id=doc_id,
                text=chunk,
                parent_doc_id=doc_id,
                metadata=metadata.copy()
            )
            self.chunked_documents.append(chunked_doc)
            
            # Prepare record for LanceDB
            vector = (self._get_embedding(chunk) if self.search_type == "vector"
                     else np.zeros(self.embedding_dim))
            
            lance_record = {
                "chunk_id": doc_id,
                "text": chunk,
                "parent_doc_id": doc_id,
                "metadata_json": json.dumps(metadata),
                "vector": vector.tolist()
            }
            lance_records.append(lance_record)
        
        # Add to LanceDB
        if lance_records:
            self.chunks_table.add(lance_records)

    def vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing matching chunks and their scores
        """
        if self.search_type != "vector":
            raise ValueError("Vector search only available when using vector search mode")
            
        query_embedding = self._get_embedding(query)
        
        # Execute search with proper parameters
        results = (self.chunks_table.search(query_embedding.tolist(), 
                                        vector_column_name="vector")
                .select(["chunk_id", "text", "parent_doc_id", "metadata_json", "_distance"])
                .limit(limit)
                .to_list())
        
        # Process results
        processed_results = []
        for result in results:
            # Convert distance to similarity score (cosine distance to similarity)
            distance = result.get('_distance', 0)
            similarity_score = 1 / (1 + distance)  # Convert distance to similarity score
            
            # Parse metadata JSON
            metadata = {}
            if result.get('metadata_json'):
                try:
                    metadata = json.loads(result.get('metadata_json', '{}'))
                except json.JSONDecodeError:
                    metadata = {}
            
            # Create processed result
            processed_result = {
                'chunk_id': result['chunk_id'],
                'text': result['text'],
                'parent_doc_id': result['parent_doc_id'],
                'metadata': metadata,
                'score': similarity_score
            }
            processed_results.append(processed_result)
        
        return processed_results

    def __len__(self) -> int:
        """Return the number of chunks in the database."""
        return len(self.chunked_documents)

    def __repr__(self) -> str:
        return (
            f"RagDatabase(search_type={self.search_type}, "
            f"default_max_segment={self.default_max_segment}, "
            f"num_documents={len(self.documents)}, "
            f"num_chunks={len(self.chunked_documents)})"
        )