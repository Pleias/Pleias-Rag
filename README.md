# Pleias RAG

Pleias RAG models are designed for Retrieval Augmented Generation (RAG) tasks. These models are optimized for structured input/output formats to ensure accurate source citation and minimize hallucinations.

This library serves the purpose of providing a simple interface to Pleias RAG models. It includes both components for creating a database of sources, searching for relevant sources and generating answers based on the sources. 

## Main Components
The two main components of the library are: 
1. **RagDatabase**: Responsible for maintaining a database of sources and searching for relevant sources based on a query.
2. **RagSystem**: Responsible for storing the database, loading the models, and generating answers based on user queries and retrieved sources.

## Usage

First, we describe the usage of the RagSystem, as it can also manage the database.

### Installation
Clone the repository and install the package using pip:
```bash
pip install .
```

### Basic Usage

```python
from Pleias_Rag.main import RagSystem

# Initialize the RAG system with an optional model path
rag_system = RagSystem(
    search_type="vector",
    db_path="data/rag_system_db",
    embeddings_model="all-MiniLM-L6-v2",
    chunk_size=300,
    model_path="meta-llama/Llama-2-7b-chat-hf"  # Optional - can also load model later
)

# Add documents to the system
rag_system.add_and_chunk_documents([
    "Neural networks are a type of machine learning model inspired by the human brain.",
    "GPT is a generative pre-trained transformer model developed by OpenAI.",
    "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models."
])

# End-to-end RAG query
query = "What are neural networks?"
result = rag_system.query(query)

# Access the results
print(f"Query: {result['query']}")
print(f"Response: {result['response']}")
```

### Loading a Model Later

```python
# Initialize without a model
rag_system = RagSystem(
    search_type="vector",
    db_path="data/rag_system_db",
    embeddings_model="all-MiniLM-L6-v2"
)

# Add documents
rag_system.add_and_chunk_documents(["Sample document text"])

# Load model when needed
rag_system.load_model("meta-llama/Llama-2-7b-chat-hf")

# Now you can generate responses
result = rag_system.query("What is the document about?")
```

### Document Management

```python
# Add documents with metadata
rag_system.add_and_chunk_documents([
    ("This is document with metadata", {"source": "website", "author": "John Doe"}),
    "This is a plain document"
])

# Add pre-chunked documents
chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
rag_system.add_pre_chunked_documents(chunks, metadata={"source": "manual chunking"})

# Get system stats
stats = rag_system.get_stats()
print(f"Documents: {stats['document_count']}")
print(f"Chunks: {stats['chunk_count']}")
print(f"Model loaded: {stats['model_loaded']}")
```

### Search and Generate Separately

```python
# Search for relevant documents
query = "What is artificial intelligence?"
results = rag_system.vector_search(query, limit=3)

# Format the results for the model
formatted_prompt = rag_system.format_for_rag_model(query, results)

# Generate a response directly from formatted prompt
response = rag_system.generate(formatted_prompt)

# Print the response
print(response)
```

## Advanced Usage

### Working with RagDatabase Directly

For more control, you can work with the RagDatabase directly:

```python
from Pleias_Rag.RagDatabase import RagDatabase, Document

# Initialize database
db = RagDatabase(
    search_type="vector",
    default_max_segment=300,
    db_path="data/lancedb", 
    embeddings_model="all-MiniLM-L6-v2"
)

# Create Document objects
doc1 = Document(text="Sample document text", metadata={"source": "research paper"})

# Add documents
db.add_and_chunk_documents([doc1, "Another document"])

# Search
results = db.vector_search("sample query", limit=5)
```

### Input/Output Format

Pleias RAG models expect input in a specific format with special tokens:

```
<|query_start|>User query<|query_end|>
<|source_start|><|source_id_start|>1<|source_id_end|>Source text 1<|source_end|>
<|source_start|><|source_id_start|>2<|source_id_end|>Source text 2<|source_end|>
<|source_analysis_start|>
```

The model will generate a response based on the provided sources.

## Generation Parameters

When initializing the RagSystem, you can configure generation parameters. It is reccomended to use the following parameters for Pleias models:

```python
rag_system = RagSystem(
    # Database parameters
    search_type="vector",
    db_path="data/rag_system_db",
    embeddings_model="all-MiniLM-L6-v2",
    chunk_size=300,
    
    # Generation parameters
    model_path="meta-llama/Llama-2-7b-chat-hf",
    max_tokens=2048,            # Maximum generated tokens
    temperature=0.0,            # Controls randomness (0.0 = deterministic)
    top_p=0.95,                 # Nucleus sampling parameter
    repitition_penalty=1.0,     # Repetition penalty
    trust_remote_code=True      # Whether to trust remote code in model repo
)
```

## Requirements

- Python 3.8+
- LanceDB for vector storage
- Sentence Transformers for embeddings
- vLLM for efficient inference


# Information about Pleias RAG Models

Pleias has developed a specialized line of language models designed specifically for Retrieval Augmented Generation (RAG). These models feature structured input/output formats to ensure accurate source citation and minimize hallucinations.

## Model Lineup

The Pleias RAG models come in different sizes to accommodate various use cases and computational requirements:

- Pleias Pico: (inlcude parameter numbers)
- Pleias Nano: (include parameter numbers)
  
## Input Format

The models accept input in the following format:

```
<|query_start|>{user question}<|query_end|>
<|source_start|><|source_id_start|>1<|source_id_end|>{source text 1}<|source_end|>
<|source_start|><|source_id_start|>2<|source_id_end|>{source text 2}<|source_end|>
<|source_analysis_start|>
```

## Output Format

The models generate output in two distinct sections:

1. Source Analysis: Following the `<|source_analysis_start|>` token, the models provide brief analyses of the provided sources in the context of the query.
2. Answer: The models then generate their responses using the following structure:

```
{source analysis}<|source_analysis_end|>
<|answer_start|>{model answer}<|answer_end|>

```

## Generation Parameters

We reccomend the following parameters for generation with vllm: 
```
SamplingParams(
           temperature=0.0,
           top_p=0.95,
           max_tokens=1200,
           repetition_penalty=1,
           stop=["#END#"],
           skip_special_tokens=False,
       )
```
Especially important is closely following the input format and keeping the temperature 0.

## Example

Input:
```
<|query_start|>{query text}<|query_end|>
<|source_start|><|source_id_start|>1<|source_id_end|>{text from source 1}<|source_end|>
<|source_start|><|source_id_start|>2<|source_id_end|>{text from source 2}<|source_end|>
<|source_analysis_start|>
```

Output:
```
{source analysis}
<|source_analysis_end|>
<|answer_start|>
{models' answers to the query}
<|answer_end|>
```

## Features

The Pleias RAG models share these core capabilities:

- Structured input/output formats for easy parsing
- Built-in source analysis capabilities
- Explicit source citation mechanisms
- Designed to minimize hallucinations
- Specialized for RAG applications

## Usage

These models' structured formats make them particularly suitable for applications requiring:

- Source-based responses
- Transparent reasoning processes
- Easy output parsing
- Reliable source attribution

