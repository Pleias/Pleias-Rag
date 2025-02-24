import re
from typing import List

def chunk_text(text: str, max_segment: int = 300) -> List[str]:
    """
    Splits text into chunks based on specified max segment size using hierarchical splitting.
    
    Args:
        text: Input text to be chunked
        max_segment: Maximum number of words per chunk (default: 300)
    
    Returns:
        List of text chunks
    """
    def main_split(text: str, max_segment: int) -> List[str]:
        segments = text.split('.\n')
        reconciled = []
        current_segment = ""
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            combined = f"{current_segment} {segment}" if current_segment else segment
            if len(combined.split()) < max_segment:
                current_segment = f"{combined}.\n" if current_segment else f"{segment}.\n"
            else:
                if current_segment:
                    reconciled.append(current_segment)
                current_segment = f"{segment}.\n"
        if current_segment:
            reconciled.append(current_segment)
        return reconciled

    def secondary_split(reconciled: List[str], max_segment: int) -> List[str]:
        max_segment += 100
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

    def tertiary_split(reconciled: List[str], max_segment: int) -> List[str]:
        max_segment += 200
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

    # Clean up text by removing multiple newlines
    text = re.sub(r" +\n", "\n", text)
    
    # Apply hierarchical splitting
    reconciled = main_split(text, max_segment)
    reconciled = secondary_split(reconciled, max_segment)
    reconciled = tertiary_split(reconciled, max_segment)
    
    # Clean up chunks by stripping whitespace
    return [chunk.strip() for chunk in reconciled]