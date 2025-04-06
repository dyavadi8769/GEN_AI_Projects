# GEN_AI_Projects

Welcome to the *GEN_AI_Projects* repository! This collection encompasses various projects that explore the capabilities and applications of Generative AI across multiple domains. Each project is organized into its respective directory, containing the necessary code, resources, and documentation to facilitate understanding and replication.

## Table of Contents

- [Project Overview](#project-overview)
  - [4.1 GenAI QA Chatbot](#41-genai-qa-chatbot)
  - [4.2 RAG Document QA](#42-rag-document-qa)
  - [4.3 RAG Conversational QA](#43-rag-conversational-qa)
  - [4.4 Search Engine](#44-search-engine)
  - [4.5 Chat with SQL DB](#45-chat-with-sql-db)
  - [4.6 YouTube Video Summarization](#46-youtube-video-summarization)
  - [4.7 Math and Data Search](#47-math-and-data-search)
  - [4.8 PDF Query RAG with AstraDB](#48-pdf-query-rag-with-astradb)
  - [4.9 Multi-Language Code Assistant](#49-multi-language-code-assistant)
  - [4.10 NVIDIA NIM](#410-nvidia-nim)
  - [4.11 CrewAI YouTube Blog](#411-crewai-youtube-blog)
  - [4.12 Hybrid Search RAG](#412-hybrid-search-rag)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

### 4.1 GenAI QA Chatbot

# GenAI QA Chatbot

*Objective*: Develop a versatile Question-Answering chatbot that supports both OpenAI and local Ollama models for context-aware responses.

*Description*: This project implements a dual-mode chatbot that can:
- Connect to OpenAI's API for cloud-based models (GPT-4 series)
- Utilize locally-hosted Ollama models (Mistral, Llama3, Gemma)
Features adjustable parameters for response customization including temperature and max tokens.

*Technologies Used*:
- Python 3.x
- Streamlit (UI framework)
- LangChain (LLM orchestration)
- OpenAI API
- Ollama (local LLMs)
- LangSmith (tracing and monitoring)

*Files*:
- `app.py`: OpenAI-only version
- `main.py`: Combined OpenAI + Ollama version

---

### 4.2 RAG Document QA (GROQ API and Llama3)


*Objective*: Implement a Retrieval-Augmented Generation (RAG) system for querying research papers using GROQ's Llama3 model.

*Description*: This project creates a document question-answering system that:
- Processes PDF research papers from a directory
- Generates vector embeddings using OpenAI
- Implements semantic search via FAISS vector store
- Provides accurate answers using Llama3-8b through GROQ API

*Technologies Used*:
- Python 3.x
- Streamlit (UI framework)
- LangChain (RAG pipeline)
- GROQ API (Llama3-8b inference)
- OpenAI Embeddings
- FAISS (vector similarity search)
- PyPDF (document loading)

*Files*:
- `app.py`: Main application file

---

### 4.3 RAG Conversational QA (with PDF Uploads and Chat History)


*Objective*: Create a conversational document Q&A system that maintains chat history context across multiple queries.

*Description*: This advanced RAG implementation features:
- PDF document processing with chat history retention
- Session-based conversation tracking
- Context-aware question reformulation
- Multi-document ingestion and vectorization
- Conversational memory using GROQ's Llama3 model

*Technologies Used*:
- Python 3.x
- Streamlit (UI framework)
- LangChain (RAG pipeline)
- GROQ API (Llama3-8b inference)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- ChromaDB (vector store)
- PyPDF (document loading)

*Key Features*:
- Multi-PDF upload interface
- Session-based chat history
- Dynamic question contextualization
- Conversational chain with memory
- Vector embeddings with chunk overlap
- Real-time chat display


---

### 4.4 Search Engine (with GROQ)

# LangChain Search Agent with GROQ

*Objective*: Create an interactive chatbot capable of performing web searches, academic paper lookups, and Wikipedia queries using GROQ's Llama3 model.

*Description*: This project implements a research assistant that:
- Performs real-time web searches (DuckDuckGo)
- Queries academic papers (Arxiv)
- Retrieves Wikipedia information
- Displays agent reasoning process
- Maintains conversational history

*Technologies Used*:
- Python 3.x
- Streamlit (UI framework)
- LangChain (agent orchestration)
- GROQ API (Llama3-8b inference)
- Search Tools:
  - DuckDuckGoSearchRun
  - ArxivQueryRun
  - WikipediaQueryRun

*Key Features*:
- Real-time search capabilities
- Thought process visualization
- Conversation history
- Multiple knowledge sources
- Streaming responses
- Error handling



---

### 4.5 Chat with SQL DB (with GROQ)


*Objective*: Create a natural language interface for querying SQL databases using GROQ's Llama3 model.

*Description*: This project implements an AI agent that:
- Connects to either SQLite or MySQL databases
- Translates natural language queries to SQL
- Executes and explains database queries
- Maintains conversation history
- Visualizes the agent's reasoning process

*Technologies Used*:
- Python 3.x
- Streamlit (UI framework)
- LangChain (SQL agent)
- GROQ API (Llama3-8b inference)
- SQLAlchemy (database connection)
- SQLite3 (embedded database)
- MySQL Connector (optional)

*Key Features*:
- Dual database support (SQLite & MySQL)
- Read-only mode for safety
- Query explanation and validation
- Cached database connections
- Real-time reasoning display
- Session-based chat history
---

### 4.6 YouTube Video Summarization


*Objective*: Create a web application that summarizes content from YouTube videos or websites using large language models.

## 🌟 Project Variants

### 1. GROQ API Version (`app.py`)
- **Model**: Gemma-7b-It via GROQ API
- **Features**:
  - YouTube video summarization
  - Website content extraction
  - 300-word concise summaries
  - URL validation

### 2. HuggingFace Version (`apphf.py`)
- **Model**: Mistral-7B-Instruct-v0.3 via HuggingFace
- **Features**:
  - Same core functionality as GROQ version
  - Customizable temperature parameter
  - Max length control

## 🛠️ Technology Stack
- **Frameworks**:
  - Streamlit (Web Interface)
  - LangChain (Orchestration)
- **Models**:
  - Gemma-7b-It (GROQ)
  - Mistral-7B-Instruct (HuggingFace)
- **Loaders**:
  - YoutubeLoader (Video transcripts)
  - UnstructuredURLLoader (Web content)

## 🚀 Key Features
| Feature | Implementation |
|---------|---------------|
| URL Validation | `validators` package |
| Content Loading | Specialized loaders for YT/Web |
| Summary Generation | Custom prompt templates |
| Error Handling | Comprehensive try-catch blocks |
| User Experience | Loading spinners, clear errors |

---

### 4.7 Math and Data Search

*Objective*: Create an AI-powered assistant that solves mathematical problems and provides research capabilities using Google's Gemma 2 model.

## 🌟 Project Overview
This application combines:
- Mathematical problem solving
- Logical reasoning
- Wikipedia research
- Step-by-step explanations

## 🚀 Key Features
| Feature | Implementation |
|---------|---------------|
| Math Calculations | LLMMathChain |
| Research Capabilities | WikipediaAPIWrapper |
| Logical Reasoning | Custom Prompt Templates |
| Interactive Chat | Streamlit UI |
| Error Handling | Automatic parsing recovery |

## 🛠️ Technology Stack
- **Core Framework**: LangChain
- **LLM**: Google Gemma2-9b-It (via GROQ)
- **Tools**:
  - Calculator (math expressions)
  - Wikipedia (research)
  - Reasoning Engine (logic problems)
- **Interface**: Streamlit

---

### 4.8 PDF Query RAG with AstraDB


*Objective*: Create a question-answering system that retrieves information from PDF documents using Astra DB's vector search capabilities.

## 🌟 Project Overview
This implementation:
- Processes PDF documents into text chunks
- Stores document embeddings in Astra DB
- Enables semantic search over document content
- Generates answers using OpenAI's LLM

## 🚀 Key Features
| Feature | Implementation |
|---------|---------------|
| PDF Processing | PyPDF2 |
| Text Chunking | CharacterTextSplitter |
| Vector Storage | Astra DB Vector Search |
| Query Interface | LangChain Index Wrapper |
| Answer Generation | OpenAI LLM |

## 🛠️ Technology Stack
- **Database**: Astra DB (Serverless Cassandra with Vector Search)
- **AI Components**:
  - OpenAI Embeddings
  - OpenAI LLM
- **Framework**: LangChain
- **PDF Processing**: PyPDF2

---

### 4.9 Multi-Language Code Assistant


*Objective*: Create a Gradio-based interface for interacting with a local CodeGuru model API endpoint.

## 🌟 Project Overview
This implementation:
- Provides a user-friendly interface for model queries
- Maintains conversation history
- Connects to a local Ollama API endpoint
- Displays responses in real-time

## 🚀 Key Features
| Feature | Implementation |
|---------|---------------|
| API Integration | `requests` library |
| Conversation History | List-based context maintenance |
| User Interface | Gradio web interface |
| Error Handling | Status code verification |
| JSON Processing | Built-in `json` module |

## 🛠️ Technology Stack
- **Core Framework**: Python 3.x
- **Web Interface**: Gradio
- **API Client**: Requests
- **Model Host**: Ollama (local)
- **Model**: codeguru

---
## 4.10 NVIDIA NIM Integration

# NVIDIA AI Projects Repository

## 1. NVIDIA API Direct Integration (`nvidia_app.py`)

*Objective*: Demonstrate direct integration with NVIDIA's API for accessing large language models.

### Key Features
- Direct API call to NVIDIA's inference endpoint
- Streams responses from Llama-3-70b model
- Simple Python implementation

---
# 4.11 - CrewAI YouTube Blog Generator


*Objective*: Automate the creation of blog content from YouTube videos using AI agents with specialized roles.

## 🌟 Project Overview
This implementation:
- Researches YouTube video content using specialized tools
- Generates blog posts from video transcripts
- Maintains sequential workflow between agents
- Caches results for efficiency

## 🚀 Key Components

### 1. AI Agents
| Agent | Role | Capabilities |
|-------|------|--------------|
| `blog_researcher` | Content Researcher | Video analysis, topic extraction |
| `blog_writer` | Content Writer | Narrative creation, simplification |

### 2. Core Features
- YouTube channel-specific content extraction
- GPT-4 powered content generation
- Sequential task execution
- Result caching
- Rate limiting (100 RPM)

## 🛠️ Technology Stack
- **Framework**: CrewAI
- **LLM**: OpenAI GPT-4-0125-preview
- **YouTube Tool**: Custom `YoutubeChannelSearchTool`
- **Environment**: Python 3.8+

```mermaid
flowchart LR
    subgraph User
        A[Input Topic] --> B[Search Videos]
    end
    
    subgraph Researcher
        B[Search Videos] --> C[Extract Content]
    end
    
    subgraph Writer
        D[Create Outline] --> E[Generate Blog]
    end
    
    subgraph Output
        F[Markdown File]
    end
    
    A -->|Trigger| B
    C -->|Research Report| D
    E --> F
    
    classDef agent fill:#e1f5fe,stroke:#039be5;
    class B,D agent;

