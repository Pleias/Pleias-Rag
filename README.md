# Pleias RAG

Pleias RAG models are designed for Retrieval Augmented Generation (RAG) tasks. These models are optimized for structured input/output formats to ensure accurate source citation and minimize hallucinations.

This library serves the purpose of providing a simple interface to Pleias RAG models. It included both components for creating a database of sources, seaching for relevant sources and generating answers based on the sources. 



# Pleias RAG Models

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

