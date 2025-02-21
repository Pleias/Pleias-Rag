from abc import ABC, abstractmethod
from typing import List, Optional
import re
from ./BaseChunker import BaseChunker

@dataclass
class BasicChunkingParams(ChunkingParams):
    """
    Parameters for basic text chunking strategy.
    
    This strategy splits text into chunks based on a maximum number of words per chunk,
    trying to maintain semantic coherence by respecting sentence boundaries where possible.
    
    Args:
        max_segment (int): Maximum number of words per chunk. Defaults to 300.
            Chunks will try to be close to but not exceed this size while maintaining
            sentence integrity where possible.
    """
    max_segment: int = 300

class BasicChunker(BaseChunker):
    """
    A chunker that splits text hierarchically based on delimiters and word counts.
    
    This chunker uses a three-level splitting strategy:
    1. First splits on paragraph breaks
    2. Then splits on sentence boundaries if chunks are still too large
    3. Finally splits on word count if necessary
    
    Args:
        max_segment (int): Target maximum number of words per chunk. Default is 300.
            Note that actual chunks might be slightly larger due to the hierarchical
            splitting strategy trying to maintain coherent text boundaries.
    """
    
    def __init__(self, max_segment: int = 300):
        self.max_segment = max_segment

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using hierarchical splitting strategy.
        
        Args:
            text (str): The text to be chunked
            
        Returns:
            List[str]: List of text chunks
        
        Example:
            >>> chunker = BasicChunker(max_segment=300)
            >>> text = "Your long text here..."
            >>> chunks = chunker.chunk_text(text)
        """
        text = re.sub(r" +\n", "\n", text)
        reconciled = self._main_split(text)
        reconciled = self._secondary_split(reconciled)
        reconciled = self._tertiary_split(reconciled)
        return reconciled

    def _main_split(self, text: str) -> List[str]:
        """
        First-level split on paragraph breaks.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Text split into primary chunks
        """
        segments = text.split('.\n')
        reconciled = []
        current_segment = ""
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            combined = f"{current_segment} {segment}" if current_segment else segment
            if len(combined.split()) < self.max_segment:
                current_segment = f"{combined}.\n" if current_segment else f"{segment}.\n"
            else:
                if current_segment:
                    reconciled.append(current_segment)
                current_segment = f"{segment}.\n"
                
        if current_segment:
            reconciled.append(current_segment)
            
        return reconciled

    def _secondary_split(self, reconciled: List[str]) -> List[str]:
        """
        Second-level split on sentence boundaries for chunks that are still too large.
        
        Args:
            reconciled (List[str]): List of primary chunks
            
        Returns:
            List[str]: Further split chunks
        """
        max_segment = self.max_segment + 100
        reconciled_secondary = []
        
        for primary_segment in reconciled:
            if len(primary_segment.split()) > max_segment:
                segments = primary_segment.split(". ")
                current_segment = ""
                
                for segment in segments:
                    segment = segment.strip()
                    if not segment:
                        continue
                        
                    combined = f"{current_segment} {segment}" if current_segment else segment
                    if len(combined.split()) < max_segment:
                        current_segment = f"{combined}. " if current_segment else f"{segment}. "
                    else:
                        if current_segment:
                            reconciled_secondary.append(current_segment)
                        current_segment = f"{segment}. "
                        
                if current_segment:
                    reconciled_secondary.append(current_segment)
            else:
                reconciled_secondary.append(primary_segment)
                
        return reconciled_secondary

    def _tertiary_split(self, reconciled: List[str]) -> List[str]:
        """
        Final-level split on word count if chunks are still too large.
        
        Args:
            reconciled (List[str]): List of secondary chunks
            
        Returns:
            List[str]: Final chunks
        """
        max_segment = self.max_segment + 200
        reconciled_tertiary = []
        
        for secondary_segment in reconciled:
            words = secondary_segment.split()
            if len(words) > max_segment:
                for i in range(0, len(words), max_segment):
                    chunk = " ".join(words[i:i + max_segment])
                    reconciled_tertiary.append(chunk)
            else:
                reconciled_tertiary.append(secondary_segment)
                
        return reconciled_tertiary