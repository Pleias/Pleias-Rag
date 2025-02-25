from typing import Dict, Any, Optional, List, Union, Tuple
import os
from vllm import LLM, SamplingParams

# Import the RagDatabase from your module
from .RagDatabase import RagDatabase, Document, ChunkedDocument

class RagSystem:
    """
    A wrapper around RagDatabase that provides direct access to the database
    functionality through passthrough methods and includes generation capabilities.
    """
    
    def __init__(
        self,
        search_type: str = "vector",
        db_path: str = "data/lancedb",
        embeddings_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        model_path: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repitition_penalty: float = 1.0,
        trust_remote_code: bool = True
    ) -> None:
        """
        Initialize the RAG system.
        
        Args:
            search_type: "vector" or "bm25" search type
            db_path: Path to store the database
            embeddings_model: Model name for sentence embeddings
            chunk_size: Default chunk size for documents
            model_path: Path to the vLLM model (optional, can be set later)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            top_p: Top-p parameter for generation
            trust_remote_code: Whether to trust remote code in model repo
        """
        self.config = {
            "search_type": search_type,
            "db_path": db_path,
            "embeddings_model": embeddings_model,
            "chunk_size": chunk_size,
            "generation": {
                "model_path": model_path,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "trust_remote_code": trust_remote_code
            }
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
        
        # Generator attributes
        self.llm = None
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        # Initialize the model if path is provided
        if model_path:
            self.load_model(model_path, trust_remote_code)
    
    def load_model(self, model_path: str, trust_remote_code: bool = True) -> None:
        """
        Load the vLLM model for generation.
        
        Args:
            model_path: Path to the model or HuggingFace model name
            trust_remote_code: Whether to trust remote code in model repo
        """
        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code
        )
        self.config["generation"]["model_path"] = model_path
        print("Model loaded successfully")
    
    
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
    
    def generate(self, formatted_prompt: str) -> str:
        """
        Generate a response from a formatted prompt with special tokens.
        
        Args:
            formatted_prompt: A prompt formatted with special tokens for RAG
            
        Returns:
            Generated text response
        """
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first or initialize with model_path.")
            
        outputs = self.llm.generate(formatted_prompt, self.sampling_params)
        
        # Extract generated text
        return outputs[0].outputs[0].text
    
    def query(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        End-to-end RAG query processing: search, format, and generate response.
        
        Args:
            query: User query string
            limit: Maximum number of search results to retrieve
            
        Returns:
            Dictionary containing query, search results, and generated response
        """
        if self.llm is None:
            raise ValueError("Model not loaded. Call load_model() first or initialize with model_path.")
        
        # Retrieve relevant documents
        search_results = self.vector_search(query, limit)
        
        # Format prompt for the model
        formatted_prompt = self.format_for_rag_model(query, search_results)
        
        # Generate response
        response = self.generate(formatted_prompt)
        
        # Return comprehensive result
        return {
            "query": query,
            "search_results": search_results,
            "formatted_prompt": formatted_prompt,
            "response": response
        }
    
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
            "search_type": self.config["search_type"],
            "model_loaded": self.llm is not None
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
        model_info = f", model={self.config['generation']['model_path']}" if self.llm else ", no model"
        return f"RagSystem(documents={len(self.db.documents)}, chunks={len(self.db.chunked_documents)}{model_info})"


# Example usage
if __name__ == "__main__":
    # Initialize the RAG system with model
    rag_system = RagSystem(
        search_type="vector",
        db_path="data/rag_system_db",
        embeddings_model="all-MiniLM-L6-v2",
        chunk_size=300,
        model_path="meta-llama/Llama-2-7b-chat-hf"  # Initialize with model
    )
    
    # Add documents
    rag_system.add_and_chunk_documents([
        "Neural networks are a type of machine learning model inspired by the human brain. They consist of interconnected nodes or 'neurons' that process and transmit information. These networks learn from examples, adapting their internal parameters through a process called training to better recognize patterns in data. Neural networks can have multiple layers (deep learning), with each layer extracting increasingly complex features. Common architectures include feedforward networks, where information flows in one direction, and recurrent neural networks (RNNs), which can process sequential data by maintaining a form of memory. Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as images, using convolution operations to detect spatial features. Neural networks have revolutionized fields like computer vision, natural language processing, and speech recognition, enabling applications from self-driving cars to voice assistants. Despite their power, neural networks can be computationally intensive and require large amounts of data for effective training.",
        
        "GPT (Generative Pre-trained Transformer) is a generative pre-trained transformer model developed by OpenAI. It represents a breakthrough in natural language processing, utilizing a transformer architecture that relies on self-attention mechanisms to understand context in text. GPT models are trained in two phases: pre-training on vast corpora of text to learn language patterns, followed by fine-tuning on specific tasks. Each iteration has grown in scale and capability—GPT-3 contains 175 billion parameters and can generate remarkably coherent and contextually appropriate text across diverse topics. GPT models excel at tasks like text completion, translation, summarization, and question answering without explicit task-specific training. They can even attempt creative writing, code generation, and logical reasoning. However, these models face challenges including potential biases inherited from training data, a tendency to generate plausible-sounding but incorrect information, and limited transparency in their decision-making processes. GPT has catalyzed discussions about AI ethics and responsible development practices in the field.",
        
        "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to enhance text generation with external knowledge. Unlike traditional language models that rely solely on parametric knowledge learned during training, RAG systems can access and integrate information from external databases or documents at generation time. The process works in two key steps: first, a retrieval component identifies relevant documents from a knowledge source based on the input query; second, a generator creates responses that incorporate both the query context and the retrieved information. This architecture offers several advantages: it provides more factual accuracy by grounding responses in authoritative sources, enables up-to-date information access beyond the model's training cutoff, allows for knowledge source customization without retraining the entire model, and offers greater transparency through citation of sources. RAG has practical applications in question answering systems, customer support automation, research assistants, and educational tools where factual accuracy and verifiability are essential. As language models continue to evolve, RAG represents an important paradigm that balances the fluency of neural text generation with the precision of information retrieval systems."
    ])
    
    # End-to-end query
    query = "What are neural networks?"
    result = rag_system.query(query)
    
    # Print results
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']}")