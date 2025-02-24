# test_rag_database.py

from src.Pleias_Rag.RagDatabase import RagDatabase, Document
import os
import shutil

def print_separator():
    print("\n" + "="*50 + "\n")

def print_search_results(query: str, results: list):
    print(f"\nVector Search Results for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Text: {result['text']}")
        print(f"Similarity Score: {result['score']:.4f}")
        if result['metadata']:
            print(f"Metadata: {result['metadata']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Parent Doc ID: {result['parent_doc_id']}")

def main():
    # Clean up any existing test database
    test_db_path = "test_data/lancedb"
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

    # Initialize database with vector search
    db = RagDatabase(
        search_type="vector",
        db_path=test_db_path,
        default_max_segment=300
    )

    # Test documents
    docs = [
        """First document, this is very long text that should be split into multiple chunks. 
        I will now describe the origins of the universe and the meaning of life. 
        This is a very important document. Life is a journey, not a destination. 
        This is the end of the first document. Just kidding, there's more text.
        
        This is a new paragraph. This is the end of the first document.""",
        
        ("Second document about databases and vector search", {"author": "John", "topic": "databases"}),
        
        Document(
            text="Third document specifically about vector embeddings and similarity search", 
            metadata={"source": "web", "topic": "embeddings"}
        )
    ]
    
    # Add documents with small chunk size to demonstrate chunking
    print("Adding and chunking documents...")
    db.add_and_chunk_documents(docs, max_segment=100)

    print("Database Summary:")
    print(db)
    print_separator()

    print("All Documents:")
    for doc in db.documents:
        print(f"\nDocument ID: {doc._id}")
        print(f"Text: {doc.text[:100]}...")  # First 100 chars
        print(f"Metadata: {doc.metadata}")
    print_separator()

    print("All Chunks:")
    for chunk in db.chunked_documents:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Parent Doc ID: {chunk.parent_doc_id}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
    print_separator()

    # Test vector search
    queries = [
        "universe and meaning of life",
        "databases and vector search",
        "embeddings and similarity"
    ]
    
    for query in queries:
        results = db.vector_search(query, limit=2)
        print_search_results(query, results)
        print_separator()

    # Test database persistence
    print("Testing Database Persistence...")
    
    # Create a new database instance pointing to the same path
    db2 = RagDatabase(
        search_type="vector",
        db_path=test_db_path,
        default_max_segment=300
    )
    
    # Verify chunks are still available
    print("\nQuerying new database instance:")
    results = db2.vector_search("universe and meaning of life", limit=1)
    if results:
        print(f"\nFound matching chunk:")
        print(f"Text: {results[0]['text']}")
        print(f"Score: {results[0]['score']:.4f}")
        print(f"Metadata: {results[0]['metadata']}")
    else:
        print("No results found")

if __name__ == "__main__":
    main()