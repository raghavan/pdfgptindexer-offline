# PDF GPT Indexer - Fully Local RAG System

A fully local Retrieval-Augmented Generation (RAG) system for indexing and querying PDF documents. **No API keys required** - everything runs locally on your machine!

## 🌟 Features

- ✅ **Fully Local**: No API keys or external services needed
- ✅ **Privacy**: All data processing happens on your machine
- ✅ **Free**: No usage costs or rate limits
- ✅ **Offline Capable**: Works without internet after initial setup
- ✅ **Easy to Use**: Simple command-line interface
- ✅ **Workshop Ready**: Perfect for educational environments

## 🏗️ Architecture

This system uses:
- **PDF Processing**: PyMuPDF for text extraction
- **Text Splitting**: LangChain's RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFace Sentence Transformers (local)
- **Vector Store**: FAISS for efficient similarity search
- **LLM**: Ollama (local LLM runner)

## 📋 Prerequisites

- Python 3.8 or higher
- At least 8GB RAM (16GB recommended)
- ~10GB free disk space (for models and dependencies)
- Internet connection for initial setup only

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd pdfgptindexer-offline
```

### Step 2: Install Python Dependencies

#### macOS

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Linux

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Windows

```powershell
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: If you encounter issues with `faiss-cpu` on Windows, you may need to install it separately:

```powershell
pip install faiss-cpu
```

### Step 3: Install Ollama

Ollama is required to run the local LLM. Follow platform-specific instructions below.

#### macOS

**Option 1: Using Homebrew (Recommended)**
```bash
brew install ollama
```

**Option 2: Manual Installation**
1. Download from [https://ollama.com/download](https://ollama.com/download)
2. Open the downloaded `.dmg` file
3. Drag Ollama to Applications folder
4. Launch Ollama from Applications

**Starting Ollama on macOS:**
- Ollama usually auto-starts after installation
- If not running, you can start it manually: `ollama serve`
- Or launch it from Applications → Ollama

#### Linux

**Installation:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Starting Ollama:**
```bash
# Start Ollama service (runs in background)
ollama serve

# Or run as a systemd service (if installed as root)
sudo systemctl enable ollama
sudo systemctl start ollama
```

#### Windows

1. Download the installer from [https://ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer (`.exe` file)
3. Ollama will start automatically after installation
4. You can verify it's running by opening: `http://localhost:11434` in your browser

**Starting Ollama on Windows:**
- Ollama runs as a Windows service and starts automatically
- If needed, you can start it from the Start Menu → Ollama

### Step 4: Download LLM Model

After Ollama is installed and running, download a model. Choose based on your needs:

**For faster, smaller models (recommended for workshops):**
```bash
ollama pull phi3
```

**For better quality (but larger/slower):**
```bash
ollama pull llama3
# or
ollama pull mistral
```

**Check available models:**
```bash
ollama list
```

**Note**: First download takes 5-15 minutes depending on your internet speed and model size. Models are cached locally after download.

### Step 5: Configure Models (Optional)

Edit the `.env` file to configure models and settings:

# Edit .env with your preferred models (optional - defaults work fine)
# OLLAMA_MODEL=phi3
# EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# TOP_K=3

See the [Configuration](#-configuration) section for details.

### Step 6: Verify Installation

**Test Ollama:**
```bash
ollama run phi3 "Hello, how are you?"
```

**Test Python dependencies:**
```bash
python -c "import langchain; import fitz; print('✓ All dependencies installed')"
```

## 📖 Usage

### Step 1: Index PDF Files

Place your PDF files in a folder (e.g., `./pdf`), then run:

```bash
python indexer.py ./pdf
```

**Options:**
- First argument: Path to PDF folder (default: `./pdf`)
- Second argument: Index output path (default: `faiss_index`)

**Example:**
```bash
# Index PDFs in ./pdf folder
python indexer.py ./pdf


**What happens:**
1. Extracts text from all PDFs in the folder
2. Splits text into chunks
3. Generates embeddings using local model (first run downloads the model)
4. Creates FAISS vector index
5. Saves index to disk

**Note**: First run may take several minutes as it downloads the embedding model (~80MB).

### Step 2: Query the Indexed Documents

```bash
python chatbot.py
```

**Options:**
- First argument: Path to index (default: `faiss_index`)

**Example:**
```bash
# Use default index
python chatbot.py

# Use custom index
python chatbot.py my_index
```

## 🔧 Configuration

### Using .env File (Recommended)

The easiest way to configure models and settings is using a `.env` file in the project root.

**Setup:**
1. Copy the example file: `cp .env.example .env`
2. Edit `.env` with your preferred settings:

```bash
# .env file
OLLAMA_MODEL=phi3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=3
```

**Available Configuration Options:**

**OLLAMA_MODEL** - Ollama LLM model name
- Options: `phi3`, `qwen2.5`, `llama3.1`, `llama3`, `mistral`, `deepseek-r1:7b`
- Default: `phi3`

**EMBEDDING_MODEL** - HuggingFace embedding model name
- Options:
  - `sentence-transformers/all-MiniLM-L6-v2` (default - fast, small, perfect for workshops)
  - `intfloat/e5-large-v2` (good balance, better quality)
  - `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (best quality)
  - `BAAI/bge-large-en-v1.5` (best for English-only)
  - `sentence-transformers/all-mpnet-base-v2` (good default)
- Default: `sentence-transformers/all-MiniLM-L6-v2`

**TOP_K** - Number of similar documents to retrieve
- Options: Any positive integer (typically 3-10)
- Default: `3`
- Higher values = more context but slower

**After changing `.env`:**
- For LLM changes: Just restart `chatbot.py`
- For embedding changes: **Re-index required** (`rm -rf faiss_index && python indexer.py ./pdf`)

### Alternative: Environment Variables

You can also set environment variables directly (useful for one-time overrides):

**macOS/Linux:**
```bash
export OLLAMA_MODEL=llama3.1
export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
export TOP_K=5
python chatbot.py
```

**Windows PowerShell:**
```powershell
$env:OLLAMA_MODEL="llama3.1"
$env:EMBEDDING_MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
$env:TOP_K="5"
python chatbot.py
```

**Windows CMD:**
```cmd
set OLLAMA_MODEL=llama3.1
set EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
set TOP_K=5
python chatbot.py
```

### Advanced Configuration

For advanced customization, you can edit the source files:

**Edit `indexer.py`** to change:
- Chunk size and overlap in `RecursiveCharacterTextSplitter`
- PDF extraction settings

## 📁 Project Structure

```
pdfgptindexer-offline/
├── indexer.py          # PDF indexing script
├── chatbot.py          # Chatbot query interface
├── requirements.txt    # Python dependencies
├── .env                # Your configuration (create from .env.example)
├── README.md          # This file
├── LICENSE            # License file
└── pdf/               # Place your PDF files here (optional)
    └── *.pdf
```

## 🐛 Troubleshooting

### Ollama Connection Issues

**Problem**: `Error: ollama server not responding`

**Solutions:**
- **macOS**: Check if Ollama is running: `ps aux | grep ollama`. If not, start it: `ollama serve` or launch from Applications
- **Linux**: Start Ollama: `ollama serve` or `sudo systemctl start ollama`
- **Windows**: Check if Ollama service is running in Services (services.msc)

**Test Ollama connection:**
```bash
curl http://localhost:11434/api/tags
```

### Model Not Found

**Problem**: `Error: Could not load model 'phi3'`

**Solution**: Pull the model first:
```bash
ollama pull phi3
```

**Check available models:**
```bash
ollama list
```

### Index Not Found

**Problem**: `Error: Index not found at 'faiss_index'`

**Solution**: Run the indexer first:
```bash
python indexer.py ./pdf
```

### Memory Issues

**Problem**: Out of memory errors during indexing

**Solutions:**
- Reduce chunk size in `indexer.py` (line 48)
- Use a smaller embedding model
- Close other applications
- Process PDFs in smaller batches

### GPU Acceleration

If you have a GPU and want faster embeddings:

**Edit `indexer.py` line 85:**
```python
model_kwargs={'device': 'cuda'}  # Change from 'cpu' to 'cuda'
```

**Edit `chatbot.py` line 24:**
```python
model_kwargs={'device': 'cuda'}  # Change from 'cpu' to 'cuda'
```

**Note**: Requires CUDA-compatible GPU and PyTorch with CUDA support.

### Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## 📊 Model Recommendations

### Embedding Models

- **sentence-transformers/all-MiniLM-L6-v2** (default): Fast, small, perfect for workshops (~80MB)
- **intfloat/e5-large-v2**: Excellent quality, good balance (~560MB)
- **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**: Best quality, multilingual (~900MB)
- **BAAI/bge-large-en-v1.5**: Best for English-only (~1.3GB)
- **sentence-transformers/all-mpnet-base-v2**: Good default (~420MB)

### LLM Models (Ollama)

- **phi3** (default): Fast, efficient, good for workshops (~3.8GB)
- **qwen2.5**: Excellent quality, great reasoning (~4.4GB)
- **llama3.1**: Best overall, improved from llama3 (~4.7GB)
- **llama3**: Good quality, balanced performance (~4.7GB)
- **mistral**: Fast and efficient (~4.1GB)
- **deepseek-r1:7b**: Best reasoning capabilities (~6.5GB)

## 🎓 Workshop Setup Tips

For conducting workshops with multiple students:

1. **Pre-download models**: Have students download models before the workshop
2. **Shared setup script**: Create a setup script for easy installation
3. **Test PDFs**: Provide sample PDFs for testing
4. **Troubleshooting guide**: Share common issues and solutions
5. **Backup plan**: Have pre-indexed FAISS indexes ready if indexing fails

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Resources

- [Ollama Documentation](https://ollama.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)

## 💡 Tips

- First indexing run downloads the embedding model (~80MB), subsequent runs are faster
- Larger PDFs take longer to index - be patient
- Keep your PDFs organized in folders for easier management
- The FAISS index can be reused - you don't need to re-index unless PDFs change
- Configure `TOP_K` in `.env` file to control how many document chunks are retrieved (more = better context but slower)
- All model configuration is done via `.env` file - no need to edit source code

---

**Happy Indexing! 🚀**
