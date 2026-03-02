# rag_question_answering_system
a Retrieval-Augmented Generation (RAG) system using FAISS and HuggingFace Transformers for context-aware question answering.
# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a basic Retrieval-Augmented Generation (RAG) system using:

- SentenceTransformers (for embeddings)
- FAISS (for vector search)
- FLAN-T5 (for answer generation)

The system retrieves relevant document chunks and generates context-aware answers.

## Architecture
User Query
→ Embedding
→ FAISS Retrieval
→ Context Injection
→ LLM Generation

## Technologies Used
- Python
- HuggingFace Transformers
- SentenceTransformers
- FAISS
- Google Colab

## Example Query
"What is a stack?"

## Future Improvements
- Add evaluation metrics
- Improve prompt engineering
- Deploy using Streamlit
