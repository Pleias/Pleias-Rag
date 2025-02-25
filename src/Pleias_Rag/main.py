from typing import Dict, Any, Optional, List, Union, Tuple
import os

# Import the RagDatabase from your module
from .RagDatabase import RagDatabase, Document, ChunkedDocument
from .Generate import Generate
class RagSystem:
    """
    A wrapper around RagDatabase that provides direct access to the database
    functionality through passthrough methods.
    """
    
    def __init__(
        self,
        search_type: str = "vector",
        db_path: str = "data/lancedb",
        embeddings_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300
    ) -> None:
        """
        Initialize the RAG system.
        
        Args:
            search_type: "vector" or "bm25" search type
            db_path: Path to store the database
            embeddings_model: Model name for sentence embeddings
            chunk_size: Default chunk size for documents
        """
        self.config = {
            "search_type": search_type,
            "db_path": db_path,
            "embeddings_model": embeddings_model,
            "chunk_size": chunk_size
        }
        
        # Create the database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize the RAG database
        self.db = RagDatabase(
            search_type=search_type,
            default_max_segment=chunk_size,
            db_path=db_path,
            embeddings_model=embeddings_model
        )
    
    # Direct passthrough methods to access database functionality
    
    def add_and_chunk_documents(
        self, 
        documents: List[Union[Document, Tuple[str, Dict[str, Any]], str]],
        max_segment: Optional[int] = None
    ) -> None:
        """Pass through to database add_and_chunk_documents method."""
        self.db.add_and_chunk_documents(documents, max_segment)
    
    def add_pre_chunked_documents(
        self,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Pass through to database add_pre_chunked_documents method."""
        self.db.add_pre_chunked_documents(chunks, metadata)
    
    def vector_search(self, query: str, limit: int = 5) -> List[Dict]:
        """Pass through to database vector_search method."""
        return self.db.vector_search(query, limit)
    
    def format_for_rag_model(self, query: str, search_results: List[Dict]) -> str:
        """
        Format search results for consumption by RAG model.
        
        Args:
            query: The user's original query
            search_results: List of search result dictionaries from vector_search
            
        Returns:
            Formatted string with query and sources in the specified format
        """
        # Initialize with the query
        formatted_output = f"<|query_start|>{query}<|query_end|>\n"
        
        # Add each source with index
        for idx, result in enumerate(search_results, 1):
            source_text = result['text']
            formatted_output += f"<|source_start|><|source_id_start|>{idx}<|source_id_end|>{source_text}<|source_end|>\n"
        
        # Add the source analysis marker
        formatted_output += "<|source_analysis_start|>"
        
        return formatted_output
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get basic statistics about the RAG system.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "config": self.config,
            "document_count": len(self.db.documents),
            "chunk_count": len(self.db.chunked_documents),
            "search_type": self.config["search_type"]
        }
    
    @property
    def documents(self) -> List[Document]:
        """Access the underlying documents list."""
        return self.db.documents
    
    @property
    def chunked_documents(self) -> List:
        """Access the underlying chunked documents list."""
        return self.db.chunked_documents
    
    def __len__(self) -> int:
        """Return the number of chunks in the database."""
        return len(self.db)
    
    def __repr__(self) -> str:
        return f"RagSystem(documents={len(self.db.documents)}, chunks={len(self.db.chunked_documents)})"


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag_system = RagSystem(
        search_type="vector",
        db_path="data/rag_system_db",
        embeddings_model="all-MiniLM-L6-v2",
        chunk_size=300
    )
    
    # Use the direct methods
    rag_system.add_and_chunk_documents(["This is a sample document about artificial intelligence.",
                                        "Machine learning is a subset of artificial intelligence that involves training models on data.",
                                        "Neural networks are a type of machine learning model inspired by the human brain."])
    
    # Search for documents
    query = "What is artificial intelligence?"
    results = rag_system.vector_search(query)
    
    # Print results
    print(f"Search results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text'][:100]}... (score: {result['score']:.4f})")
    
    # Format results for RAG model
    formatted_results = rag_system.format_for_rag_model(query, results)
    print("\nFormatted for RAG model:")
    print(formatted_results)
    
    # Print system stats
    print("\nSystem stats:")
    stats = rag_system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Show document count using property
    print(f"\nTotal documents: {len(rag_system.documents)}")
    print(f"Total chunks: {len(rag_system.chunked_documents)}")