# hindu-scripture-chatbot
Ask questions about Hindu scriptures and get thoughtful, context-aware answers. Powered by AI with RAG, covering Gita, Ramayana, Mahabharata &amp; Vedas.



# 🕉️ Hindu Scripture Chatbot

A production-grade AI chatbot that provides thoughtful, context-aware answers about Hindu scriptures including the Bhagavad Gita, Mahabharata, Ramayana, and Vedas. Built with advanced RAG (Retrieval-Augmented Generation), conversation memory, and comprehensive safety layers.

## ✨ Features

### Core Capabilities
- **Multi-Scripture Knowledge**: Answers questions from Bhagavad Gita, Mahabharata, Ramayana, and all four Vedas
- **Intelligent Classification**: LLM-powered scripture detection with context awareness
- **RAG-Powered Responses**: Vector search using Supabase + BGE embeddings for accurate, scripture-grounded answers
- **Conversation Memory**: Maintains context across conversations, handles follow-up questions and pronouns
- **Multi-Scripture Synthesis**: Compares and synthesizes teachings across multiple scriptures

### Safety & Sensitivity
- **Content Moderation**: OpenAI moderation API integration for user safety
- **Sensitive Topic Handling**: Respectful responses to comparative, historical, and conversion-related questions
- **Crisis Support**: Appropriate handling of mental health queries with helpline resources
- **Out-of-Domain Detection**: Politely declines non-scripture topics

### User Experience
- **Context-Aware**: Understands pronouns, follow-ups, and conversational refinements
- **Warm & Respectful Tone**: Addresses users as "Dear seeker" with compassionate guidance
- **Meta Guidance**: Recommends which scripture to start with for beginners
- **Session Management**: Supports both stateless (Colab) and stateful (API) conversations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Safety & Moderation Layer                       │
│  • OpenAI Moderation API                                     │
│  • Sensitive Topic Detection                                 │
│  • Crisis Intervention                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           LLM-Based Scripture Classification                 │
│  • GPT-4o-mini classifier                                    │
│  • Context-aware (uses conversation history)                 │
│  • Handles pronouns, typos, follow-ups                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  RAG Retrieval Layer                         │
│  • Query embedding (BAAI/bge-large-en-v1.5)                 │
│  • Supabase vector search                                    │
│  • Scripture-specific filtering                              │
│  • Top-K context retrieval                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Answer Generation (GPT-4o-mini)                 │
│  • Context-aware prompting                                   │
│  • Memory integration                                        │
│  • Multi-scripture synthesis                                 │
│  • Respectful, grounded responses                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Final Response                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
hindu-scripture-chatbot/
├── chatbot.py          # Core chatbot logic (all functions)
├── api.py              # FastAPI endpoint wrapper
├── schemas.py          # Pydantic request/response models
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not in git)
├── .env.example       # Template for environment variables
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Supabase account with vector database setup
- HuggingFace token (for embedding model)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/hindu-scripture-chatbot.git
cd hindu-scripture-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

Required environment variables:
```env
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Database Configuration
TABLE_NAME=your_table_name
TEXT_COLUMN=text
ID_COLUMN=id
BATCH_SIZE=100

# HuggingFace (if needed)
HUGGINGFACE_TOKEN=your_hf_token
```

5. **Run the chatbot**


##  Acknowledgments

- **OpenAI** - GPT-4o-mini for classification and generation
- **Supabase** - Vector database and hosting
- **Sentence Transformers** - BGE embedding model
- **Hindu scriptures** - Bhagavad Gita, Mahabharata, Ramayana, Vedas


---
