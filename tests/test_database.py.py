import unittest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from Pleias_Rag import RagDatabase, Document, ChunkingStrategy, SearchType

# Rest of your test code remains the same...
class TestRagDatabase(unittest.TestCase):
    def setUp(self):
        """Set up a database instance before each test."""
        self.db = RagDatabase()
        
        # Sample texts for testing
        self.short_text = "This is a short document."
        self.long_text = """
        This is a longer document that should be split into multiple chunks.
        It has multiple sentences and should demonstrate the chunking behavior.
        We'll add even more text to ensure we get multiple chunks.
        Here's another sentence to make it longer.
        And another one just to be sure we have enough text.
        """
        self.sample_metadata = {"author": "Test Author", "date": "2024"}

    def test_document_addition(self):
        """Test different ways of adding documents."""
        # Test adding a simple string
        self.db.add_documents([self.short_text])
        self.assertEqual(len(self.db.documents), 1)
        self.assertEqual(self.db.documents[0].text, self.short_text)
        
        # Test adding with metadata
        self.db.add_documents([(self.short_text, self.sample_metadata)])
        self.assertEqual(len(self.db.documents), 2)
        self.assertEqual(self.db.documents[1].metadata, self.sample_metadata)
        
        # Test adding a Document object
        doc = Document(text=self.short_text, metadata=self.sample_metadata)
        self.db.add_documents([doc])
        self.assertEqual(len(self.db.documents), 3)

    def test_document_ids(self):
        """Test automatic ID generation."""
        self.db.add_documents([self.short_text, self.long_text])
        
        # Check if IDs are unique and properly formatted
        self.assertEqual(self.db.documents[0]._id, "doc_0")
        self.assertEqual(self.db.documents[1]._id, "doc_1")

    def test_chunking(self):
        """Test document chunking behavior."""
        # Test with basic chunking
        self.db.add_documents([self.long_text])
        self.assertGreater(len(self.db.chunked_documents), 1)
        
        # Test with no chunking
        db_no_chunk = RagDatabase(chunking_strategy=ChunkingStrategy.NONE)
        db_no_chunk.add_documents([self.long_text])
        self.assertEqual(len(db_no_chunk.chunked_documents), 1)

    def test_chunk_metadata(self):
        """Test that chunks inherit document metadata."""
        doc_with_metadata = (self.long_text, self.sample_metadata)
        self.db.add_documents([doc_with_metadata])
        
        # Check if all chunks have the same metadata as parent
        for chunk in self.db.chunked_documents:
            self.assertEqual(chunk.metadata, self.sample_metadata)

    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            self.db.add_documents([123])  # Invalid type
        
        with self.assertRaises(ValueError):
            self.db.add_documents([(self.short_text, self.sample_metadata, "extra")])  # Too many tuple elements

    def test_empty_document(self):
        """Test handling of empty documents."""
        self.db.add_documents([""])
        self.assertEqual(len(self.db.documents), 1)
        self.assertEqual(len(self.db.chunked_documents), 1)

if __name__ == '__main__':
    unittest.main()