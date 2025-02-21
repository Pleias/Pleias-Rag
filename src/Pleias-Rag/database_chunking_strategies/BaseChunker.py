from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ChunkingParams(ABC):
    """
    Abstract base class for chunking parameters.
    
    This class serves as a type-safe base for different chunking strategy parameters.
    Each chunking strategy should have its own parameter class inheriting from this.
    """
    pass


class BaseChunker(ABC):
    """
    Abstract base class for text chunking strategies.
    
    This class defines the interface that all chunking strategies must implement.
    """
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks according to the strategy's logic.
        
        Args:
            text (str): The text to be chunked
            
        Returns:
            List[str]: List of text chunks
        """
        pass
