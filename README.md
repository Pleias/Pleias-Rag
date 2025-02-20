# Pleias Pico

Pleias Pico is a specialized language model developed by Pleias, designed specifically for Retrieval Augmented Generation (RAG). The model features a structured input/output format to ensure accurate source citation and minimize hallucinations.

## Input Format

The model accepts input in the following format:

```
<|query_start|>{user question}<|query_end|>
<|source_start|><|source_id_start|>1<|source_id_end|>{source text 1}<|source_end|>
<|source_start|><|source_id_start|>2<|source_id_end|>{source text 2}<|source_end|>
<|source_analysis_start|>
```

## Output Format

The model generates output in two distinct sections:

1. Source Analysis: Following the `<|source_analysis_start|>` token, the model provides a brief analysis of the provided sources in the context of the query.

2. Answer: The model then generates its response using the following structure:
```
{source analysis}<|source_analysis_end|>
<|answer_start|>{model answer}<|answer_end|>
```

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
{model's answer to the query}
<|answer_end|>
```

## Features
- Structured input/output format for easy parsing
- Built-in source analysis
- Explicit source citation
- Designed to minimize hallucinations
- Specialized for RAG applications

## Usage

The model's structured format makes it particularly suitable for applications requiring:
- Source-based responses
- Transparent reasoning
- Easy output parsing
- Reliable source attribution
