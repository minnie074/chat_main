
#### 1. **Web Scraper** (`src/scraper.py`)
- Uses `requests` and `BeautifulSoup4` for HTML parsing
- Removes irrelevant sections (nav, footer, ads, scripts, styles)
- Implements robust error handling for:
  - Invalid URLs
  - Unreachable websites
  - Timeout errors
  - HTTP errors (404, 403, etc.)
  - Empty content
- Validates content sufficiency before proceeding

#### 2. **Text Chunker** (`src/chunker.py`)
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Configurable chunk size (default: 800 characters)
- Configurable overlap (default: 200 characters)
- Semantic splitting: prioritizes paragraphs > sentences > words
- Removes duplicate chunks
- Preserves metadata (URL, title, chunk ID, etc.)

#### 3. **Vector Store** (`src/vector_store.py`)
- **Embedding Model**: `text-embedding-3-small` (OpenAI)
  - Cost-effective
  - High quality (1536 dimensions)
  - Fast inference
- **Vector Database**: ChromaDB
  - Lightweight and local
  - Persistent storage
  - No external dependencies
  - Fast similarity search
- Stores embeddings with metadata
- Supports loading/saving from disk

#### 4. **QA Chain** (`src/qa_chain.py`)
- **LLM Model**: GPT-3.5-Turbo
  - Fast and cost-effective
  - Good balance of performance/cost
  - Reliable instruction following
- Temperature: 0.0 (deterministic responses)
- Implements `ConversationalRetrievalChain` from LangChain
- Maintains conversation history with `ConversationBufferMemory`
- Custom prompt template enforcing strict grounding

#### 5. **Chatbot Orchestrator** (`src/chatbot.py`)
- Coordinates all components
- Manages application state
- Handles the complete pipeline:
  1. Scrape → 2. Chunk → 3. Embed → 4. Store → 5. Answer

## 🧠 LLM & Framework Choices

### Why GPT-3.5-Turbo?

1. **Cost-Effective**: ~10x cheaper than GPT-4 for similar tasks
2. **Fast Response**: Low latency for better user experience
3. **Sufficient Capability**: Excellent for retrieval-augmented generation
4. **Reliable**: Consistent performance with clear prompts

### Why LangChain?

1. **Production-Ready**: Battle-tested framework with robust components
2. **Modular Design**: Easy to swap components (LLM, vector store, etc.)
3. **Built-in Memory**: ConversationBufferMemory for context management
4. **RAG Support**: Excellent retrieval-augmented generation tools
5. **Active Community**: Regular updates and extensive documentation

### Why Text-Embedding-3-Small?

1. **Performance**: High quality embeddings at 1536 dimensions
2. **Cost**: 5x cheaper than text-embedding-3-large
3. **Speed**: Fast embedding generation
4. **OpenAI Native**: Seamless integration with OpenAI ecosystem

### Why ChromaDB?

1. **Simplicity**: Easy to set up, no external servers required
2. **Persistence**: Local storage with automatic persistence
3. **Performance**: Fast similarity search for small-medium datasets
4. **Free**: No costs for local usage
5. **Python-Native**: Excellent Python integration

**Alternatives Considered:**
- **Pinecone**: Requires external service, costs money
- **Weaviate**: More complex setup, overkill for single-website use
- **FAISS**: No built-in persistence, requires manual management
- **Qdrant**: More features than needed, heavier installation

## 📦 Installation & Setup

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd website-chatbot
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key (required)

**Optional Configuration:**
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-3-small)
- `LLM_MODEL`: Language model (default: gpt-3.5-turbo)
- `CHUNK_SIZE`: Chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 4)

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 🚀 Usage Guide

### 1. Index a Website

1. Enter a valid HTTP/HTTPS URL in the sidebar
2. Click "🔍 Index Website"
3. Wait for the indexing process to complete (10-30 seconds)
4. View statistics: chunks created, word count, etc.

**Example URLs to Try:**
- `https://en.wikipedia.org/wiki/Artificial_intelligence`
- `https://www.anthropic.com`
- `https://python.org`

### 2. Ask Questions

1. Type your question in the input box
2. Click "📤 Send Question"
3. View the AI-generated answer
4. Check sources by expanding "📚 View Sources"

**Example Questions:**
- "What is the main topic of this website?"
- "Can you summarize the key points?"
- "What are the benefits mentioned?"
- "Are there any statistics or numbers?"

### 3. Conversation Features

- **Context Maintained**: Follow-up questions use conversation history
- **Clear Conversation**: Reset conversation memory without re-indexing
- **View Sources**: See which content chunks were used
- **Reset All**: Clear everything and start fresh

## 🎯 Key Features

### ✅ Strict Grounding

The chatbot ONLY answers from website content. If information isn't available, it responds:

> "The answer is not available on the provided website."

This is enforced through:
1. Custom prompt template with strict instructions
2. Temperature 0.0 for deterministic responses
3. Retrieved context explicitly shown in prompt

### ✅ Conversation Memory

- Maintains short-term memory within a session
- Uses LangChain's `ConversationBufferMemory`
- Context from previous questions improves follow-up answers
- Memory clears when:
  - "Clear Conversation" button clicked
  - New website indexed
  - Application reset

### ✅ Robust Error Handling

| Error Type | Handling |
|------------|----------|
| Invalid URL | Validation before processing |
| Unreachable Website | Timeout with clear error message |
| Empty Content | Minimum content check (100 chars) |
| HTTP Errors | Specific messages (404, 403, etc.) |
| API Failures | Graceful degradation with error messages |

### ✅ Content Processing

**Removed Elements:**
- Navigation menus (`<nav>`)
- Headers (`<header>`)
- Footers (`<footer>`)
- Advertisements (by class/id patterns)
- Scripts (`<script>`)
- Styles (`<style>`)
- Iframes, comments, etc.

**Duplicate Removal:**
- Content-based deduplication
- Ignores very short chunks (<50 chars)

## 📁 Project Structure

```
website-chatbot/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── src/                       # Source code modules
│   ├── scraper.py            # Web scraping & content extraction
│   ├── chunker.py            # Text processing & chunking
│   ├── vector_store.py       # Embeddings & ChromaDB operations
│   ├── qa_chain.py           # QA logic & conversation memory
│   └── chatbot.py            # Main orchestrator
│
├── chroma_db/                # ChromaDB persistence (auto-created)
└── logs/                     # Application logs (auto-created)
```

## 🔧 Configuration Options

### Chunking Strategy

```python
# Modify in .env
CHUNK_SIZE=800          # Characters per chunk
CHUNK_OVERLAP=200       # Overlap between chunks
```

**Guidelines:**
- Larger chunks: Better context, slower retrieval
- Smaller chunks: Faster retrieval, may miss context
- Overlap: Prevents information loss at boundaries

### Retrieval Strategy

```python
# Modify in .env
TOP_K_RESULTS=4         # Number of chunks to retrieve
```

**Guidelines:**
- More results: Better recall, more noise
- Fewer results: Faster, may miss relevant info
- Sweet spot: 3-5 for most use cases

### LLM Parameters

```python
# Modify in .env
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.0     # Deterministic (recommended)
LLM_MAX_TOKENS=1000     # Response length limit
```

## ⚠️ Limitations & Assumptions

### Current Limitations

1. **Single Page Only**: Doesn't crawl multiple pages or follow links
2. **Text-Only**: Doesn't process images, videos, or PDFs
3. **No Authentication**: Can't access pages behind login
4. **JavaScript-Heavy Sites**: May not work well with SPAs
5. **Rate Limiting**: No built-in rate limiting for API calls
6. **Memory Size**: Conversation memory grows unbounded in session

### Assumptions

1. Website content is primarily text-based HTML
2. Content is publicly accessible
3. User has valid OpenAI API key with credits
4. Python 3.9+ environment available
5. Sufficient disk space for ChromaDB persistence

### Known Issues

- Very large websites (>10,000 words) may take longer to index
- Some websites block automated scraping (403 errors)
- Dynamic content loaded by JavaScript won't be captured

## 🚀 Future Improvements

### Short-Term
- [ ] Add support for multiple pages/links
- [ ] Implement PDF document support
- [ ] Add rate limiting for API calls
- [ ] Support for alternative LLMs (Claude, Llama, etc.)
- [ ] Improve JavaScript-heavy site handling

### Long-Term
- [ ] Multi-website indexing and comparison
- [ ] Export conversation history
- [ ] Custom embedding models (open-source)
- [ ] Advanced filtering and search
- [ ] Analytics dashboard
- [ ] API endpoint for programmatic access

## 🧪 Testing

### Manual Testing Checklist

- [ ] Valid URL indexing
- [ ] Invalid URL handling
- [ ] Empty content handling
- [ ] Unreachable website handling
- [ ] Question answering accuracy
- [ ] "Not available" responses for out-of-scope questions
- [ ] Conversation memory functionality
- [ ] Source attribution
- [ ] Reset functionality

### Test URLs

```python
# Good test cases
https://en.wikipedia.org/wiki/Python_(programming_language)
https://www.anthropic.com
https://docs.python.org/3/

# Error test cases
http://this-does-not-exist-12345.com  # Unreachable
https://example.com/404                # 404 error
```

## 📊 Performance Metrics

Typical performance on a modern machine:

| Metric | Value |
|--------|-------|
| Indexing Time | 10-30 seconds |
| Question Response | 2-5 seconds |
| Memory Usage | ~200-500 MB |
| Chunks per 1000 words | ~3-4 chunks |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **OpenAI**: For GPT-3.5 and embedding models
- **ChromaDB**: For the simple yet powerful vector database
- **Streamlit**: For the amazing web framework

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the code comments
