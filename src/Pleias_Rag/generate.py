from vllm import LLM, SamplingParams

class RAGGenerator:
    """
    A simplified RAG Generator using vLLM that works with formatted queries.
    This generator assumes the RagSystem will format the prompts with appropriate 
    query and source markers.
    """
    
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        repitition_penalty: float = 1.0,
        trust_remote_code: bool = True
    ):
        """
        Initialize the RAG Generator with vLLM.
        
        Args:
            model_path: Path to the model or HuggingFace model name
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Top-p sampling parameter (lower = more focused)
            trust_remote_code: Whether to trust remote code in model repo
        """
        self.model_path = model_path
        
        # Initialize vLLM model
        print(f"Loading model from {model_path}...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=trust_remote_code
        )
        print("Model loaded successfully")
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens, 
            repitition_penalty=repitition_penalty
        )
    
    def generate(self, formatted_prompt: str) -> str:
        """
        Generate a response from a formatted prompt with special tokens.
        
        Args:
            formatted_prompt: A prompt formatted with special tokens for RAG
            
        Returns:
            Generated text response
        """
        outputs = self.llm.generate(formatted_prompt, self.sampling_params)
        
        # Extract generated text
        return outputs[0].outputs[0].text