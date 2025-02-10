# MedChat AI

A portfolio project exploring the implementation of a healthcare chatbot using modern AI technologies. This project demonstrates the practical application of large language models, vector databases, and the LangChain framework.

## Learning Objectives

This project was built to gain hands-on experience with:
- Building a retrieval-augmented generation (RAG) system using LangChain
- Setting up and using vector databases (ChromaDB) for efficient similarity search
- Working with embeddings using HuggingFace's sentence transformers
- Integrating OpenAI's GPT models via API
- Implementing hybrid search approaches combining vector similarity and LLM capabilities

## Technical Implementation

### Key Components
- **Vector Store**: ChromaDB for storing and retrieving document embeddings
- **Embeddings**: HuggingFace's sentence-transformers/all-MiniLM-L6-v2
- **Language Model**: OpenAI's GPT-3.5 Turbo
- **Framework**: LangChain for building the AI application pipeline

### Core Features
- Vector similarity search for relevant medical information
- Fallback to direct LLM responses when no relevant context is found
- Processing and storing of healthcare Q&A data
- Interactive command-line interface for testing

## Implementation Details

### Data Processing Pipeline
1. Raw data loading from CSV
2. Text preprocessing and formatting
3. Train/validation split
4. Conversion to vector embeddings

### Chatbot Architecture
- **Vector Search**: First attempts to find relevant information using cosine similarity
- **Context Integration**: Combines retrieved context with the user's question
- **LLM Processing**: Uses GPT-3.5 Turbo to generate natural language responses
- **Fallback Mechanism**: Direct LLM responses when no relevant context is found

## Technologies Used

- Python 3.8+
- LangChain
- ChromaDB
- HuggingFace Transformers
- OpenAI API
- pandas & scikit-learn (for data processing)

## Future Learning Opportunities

Areas for potential expansion:
- Implementing additional embedding models
- Exploring different vector database solutions
- Adding memory capabilities for multi-turn conversations
- Implementing streaming responses
- Adding evaluation metrics

## Requirements

- OpenAI API key
- Python 3.8 or higher
- See requirements.txt for Python package dependencies

## Acknowledgments

This project was built for learning purposes and leverages several open-source tools and frameworks:
- LangChain for the RAG implementation
- HuggingFace for embeddings
- ChromaDB for vector storage
- OpenAI for language model capabilities