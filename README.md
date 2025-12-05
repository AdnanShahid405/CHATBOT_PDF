# PDF Chatbot - AI-Powered Document Q&A System

An intelligent chatbot application that enables users to upload PDF documents and ask questions about their content using advanced AI and Natural Language Processing techniques.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that processes PDF documents and answers user questions based on the document content. The system uses semantic search to find relevant information and Google's Gemini AI to generate accurate, contextual responses.

**Built for:** AppLab AI/ML Engineer Position Assignment

---

## ✨ Features

### Core Functionality
- 📄 **PDF Document Upload** - Supports both text-based and scanned PDFs
- 🤖 **Intelligent Q&A** - AI-powered responses based on document content
- 🔍 **Semantic Search** - Uses embeddings for accurate context retrieval
- 📊 **Document Processing** - Smart text chunking with overlap for better context
- 🖼️ **OCR Support** - Extracts text from scanned/image-based PDFs using Tesseract

### Technical Features
- ⚡ **REST API** - Clean API endpoints for upload and chat
- 🎨 **Modern UI** - Responsive web interface with drag-and-drop
- 🐳 **Docker Ready** - Containerized for easy deployment
- 📝 **Source Attribution** - Shows which document sections were used
- 🔄 **Retry Logic** - Automatic API retry with exponential backoff
- 📈 **Real-time Stats** - Document statistics and processing status

---

## 🏗️ Architecture
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Frontend  │────────▶│  Flask API   │────────▶│   Gemini    │
│   (HTML/JS) │         │   (Python)   │         │     AI      │
└─────────────┘         └──────────────┘         └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Sentence    │
                        │ Transformers │
                        │  (Embeddings)│
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │   Vector     │
                        │   Storage    │
                        │  (In-Memory) │
                        └──────────────┘
```

**Workflow:**
1. User uploads PDF → Text extraction (PyPDF2 or OCR)
2. Text chunked into segments with overlap
3. Chunks converted to embeddings (sentence-transformers)
4. User asks question → Query converted to embedding
5. Cosine similarity finds relevant chunks (Top-K retrieval)
6. Gemini AI generates answer using retrieved context

---

## 🛠️ Technologies Used

### Backend
- **Flask** - Web framework
- **Google Gemini AI** - Answer generation (gemini-2.5-flash model)
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **PyPDF2** - PDF text extraction
- **Tesseract OCR** - Scanned PDF processing
- **NumPy** - Vector operations and similarity calculations

### Frontend
- **HTML5/CSS3** - UI structure and styling
- **JavaScript** - Interactive functionality
- **Font Awesome** - Icons

### Deployment
- **Docker** - Containerization
- **Python 3.10+** - Runtime environment

---

## 📦 Prerequisites

### Required
- Python 3.10 or higher
- pip (Python package manager)
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))

### Optional (for OCR support)
- Tesseract OCR ([Installation guide](https://github.com/tesseract-ocr/tesseract))
- Poppler (for pdf2image)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install OCR Dependencies (Optional)

**Windows:**
1. Download Tesseract installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to `C:\Program Files\Tesseract-OCR\`
3. Download Poppler from [here](https://github.com/oschwartz10612/poppler-windows/releases/)
4. Extract and add to PATH

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

---

## ⚙️ Configuration

### 1. Set Up Environment Variables

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
FLASK_ENV=development
```

### 2. Update Configuration (Optional)

Edit `app.py` to customize settings:
```python
# Model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Chunking settings
CHUNK_SIZE = 200          # Words per chunk
CHUNK_OVERLAP = 30        # Overlapping words
TOP_K_RESULTS = 5         # Number of chunks to retrieve
```

---

## 💻 Usage

### Running Locally

1. **Start the Application:**
```bash
python app.py
```

2. **Access the Interface:**
   - Open browser to: `http://localhost:8000`

3. **Upload a Document:**
   - Click "Browse Files" or drag & drop a PDF
   - Wait for processing confirmation

4. **Ask Questions:**
   - Type your question in the chat input
   - Press Enter or click send
   - Receive AI-generated answers with sources

### Example Questions
```
"What is the main topic of this document?"
"Summarize the key findings in section 3"
"What recommendations are mentioned?"
"Explain the methodology used"
```

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Endpoints

#### 1. Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "running",
  "message": "PDF Chatbot API is running",
  "documents_loaded": true,
  "chunks_count": 150,
  "models_initialized": true,
  "gemini_model": "gemini-2.5-flash"
}
```

---

#### 2. Upload Document
```http
POST /api/upload
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (PDF file)

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

**Success Response (200):**
```json
{
  "message": "Document uploaded and processed successfully",
  "filename": "document.pdf",
  "chunks_created": 150,
  "total_characters": 45000
}
```

**Error Response (400):**
```json
{
  "error": "Only PDF files are supported"
}
```

---

#### 3. Chat
```http
POST /api/chat
```

**Request:**
```json
{
  "question": "What is the main conclusion?"
}
```

**Example (cURL):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion?"}'
```

**Success Response (200):**
```json
{
  "answer": "According to Source 1, the main conclusion is...",
  "sources": [
    {
      "text": "Excerpt from the document...",
      "similarity": 0.8542,
      "metadata": {
        "filename": "document.pdf",
        "chunk_id": 42
      }
    }
  ]
}
```

**Error Response (400):**
```json
{
  "error": "No documents uploaded. Please upload a PDF document first."
}
```

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t pdf-chatbot .
```

### Run the Container
```bash
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  pdf-chatbot
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  pdf-chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./temp_ocr:/app/temp_ocr
```

Run with:
```bash
docker-compose up
```

---

## 📁 Project Structure
```
pdf-chatbot/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose config
├── .env                        # Environment variables (not in git)
├── .env.example               # Example environment file
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
│
├── templates/
│   └── index.html             # Main HTML template
│
├── static/
│   ├── css/
│   │   └── style.css          # Styling
│   └── js/
│       └── main.js            # Frontend JavaScript
│
└── temp_ocr/                  # Temporary OCR files (auto-created)
```

---

## 🔬 How It Works

### 1. Document Processing Pipeline
```python
PDF Upload → Text Extraction → Chunking → Embedding Generation → Storage
```

**Text Extraction:**
- Detects if PDF is text-based or scanned
- Uses PyPDF2 for text PDFs
- Uses Tesseract OCR for scanned PDFs

**Chunking Strategy:**
- Splits text into 200-word chunks
- 30-word overlap between chunks
- Preserves context across boundaries

### 2. Question Answering Pipeline
```python
User Question → Query Embedding → Similarity Search → Context Retrieval → AI Generation
```

**Semantic Search:**
- Converts query to embedding vector
- Calculates cosine similarity with all chunks
- Retrieves top 5 most relevant chunks

**Answer Generation:**
- Sends relevant context to Gemini AI
- Prompts for source-attributed answers
- Returns structured response with citations

### 3. Key Algorithms

**Cosine Similarity:**
```python
similarity = dot(query_vector, chunk_vector) / 
             (norm(query_vector) * norm(chunk_vector))
```

**Chunk Overlap:**
```python
chunk[i] = words[i : i + chunk_size]
i += (chunk_size - overlap)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Tesseract not found" Error
```bash
# Windows: Update path in app.py
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux/Mac: Install via package manager
sudo apt-get install tesseract-ocr  # Ubuntu
brew install tesseract              # macOS
```

#### 2. "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

#### 3. "Gemini API Error"
- Verify API key is correct
- Check API quota at [Google AI Studio](https://makersuite.google.com/)
- Ensure internet connection is active

#### 4. "Empty response from PDF"
- Check if PDF is encrypted
- Try with OCR-enabled processing
- Verify PDF isn't corrupted

#### 5. Port Already in Use
```bash
# Change port in app.py
app.run(host="0.0.0.0", port=8001, debug=True)
```

---

## 🧪 Testing

### Manual Testing
1. Upload a sample PDF
2. Ask questions about the content
3. Verify answers match document content
4. Check source attribution

### API Testing with cURL
```bash
# Health check
curl http://localhost:8000/api/health

# Upload document
curl -X POST http://localhost:8000/api/upload \
  -F "file=@test.pdf"

# Ask question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'
```

---

## 🎨 Customization

### Changing AI Model
```python
# In app.py
GEMINI_MODEL_NAME = "gemini-pro"  # or other Gemini models
```

### Adjusting Chunk Size
```python
CHUNK_SIZE = 300      # Larger chunks = more context per chunk
CHUNK_OVERLAP = 50    # More overlap = better continuity
```

### Increasing Retrieved Context
```python
TOP_K_RESULTS = 10    # Retrieve more chunks per query
```

---

## 🚀 Future Enhancements

- [ ] Multi-document support
- [ ] Conversation history persistence
- [ ] User authentication
- [ ] Support for DOCX, TXT files
- [ ] Vector database integration (Pinecone/Chroma)
- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Export chat history
- [ ] Advanced filtering options

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@AdnanShahid405](https://github.com/AdnanShahid405)
- Email: adnanshahid405@gmail.com
- LinkedIn: [Your Profile](https://www.linkedin.com/in/muhammad-adnan-shahid-364019222/)

---

## 🙏 Acknowledgments

- [Google Gemini AI](https://ai.google.dev/) for the language model
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- [Flask](https://flask.palletsprojects.com/) for the web framework

---

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub
3. Contact via email

---

## 📊 Performance Notes

- **Embedding Model**: ~85MB download on first run
- **Processing Time**: ~2-5 seconds per PDF page
- **Memory Usage**: ~500MB base + document size
- **API Latency**: ~1-3 seconds per query (depends on Gemini API)

---

**Made with ❤️ for AppLab AI/ML Engineer Assignment**
