from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import logging
import io
import time
import os
from dotenv import load_dotenv
load_dotenv()
# Configure logging FIRST (before using logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OCR imports (optional - only used if PDF is scanned)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
    
    # Set Tesseract path for Windows (adjust if installed elsewhere)
    if os.name == 'nt':  # Windows
        tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            logger.warning("Tesseract not found at default path. OCR may not work.")
    
    logger.info("✓ OCR capabilities available (pytesseract + pdf2image)")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("⚠️ OCR libraries not installed. Scanned PDFs won't be processed.")
    logger.warning("Install with: pip install pytesseract pdf2image pillow")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
GEMINI_API_KEY = os.getenv("GEMINI_API")

# Model settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Chunking settings
CHUNK_SIZE = 200  # Reduced for better precision (was 500)
CHUNK_OVERLAP = 30  # Proportional overlap (was 50)
TOP_K_RESULTS = 5  # Increased to get more context (was 3)

# ==================== GLOBAL STORAGE ====================
document_store = {
    "chunks": [],
    "embeddings": [],
    "metadata": []
}

# Model instances
embedding_model = None
gemini_model = None

# ==================== INITIALIZATION ====================
def initialize_models():
    """Initialize the embedding model and Gemini API"""
    global embedding_model, gemini_model
    
    try:
        logger.info("="*50)
        logger.info("Initializing models...")
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("✓ Embedding model loaded successfully")
        
        logger.info("Configuring Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"✓ Gemini API configured successfully with model: {GEMINI_MODEL_NAME}")
        
        logger.info("="*50)
        logger.info("All models initialized successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise

# ==================== PDF PROCESSING ====================
# ==================== PDF PROCESSING ====================
def is_pdf_text_based(pdf_bytes):
    """
    Check if PDF contains extractable text or is image-based (scanned)
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        bool: True if text-based, False if image-based/scanned
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        # Sample first 3 pages (or all if less than 3)
        pages_to_check = min(3, len(pdf_reader.pages))
        total_text_length = 0
        
        for i in range(pages_to_check):
            page = pdf_reader.pages[i]
            text = page.extract_text()
            total_text_length += len(text.strip())
        
        # Heuristic: If we get less than 50 characters per page on average, it's likely scanned
        avg_chars_per_page = total_text_length / pages_to_check
        
        logger.info(f"PDF Analysis: {avg_chars_per_page:.0f} avg characters per page")
        
        if avg_chars_per_page < 50:
            logger.info("📄 Detection: Image-based/Scanned PDF (will use OCR)")
            return False
        else:
            logger.info("📄 Detection: Text-based PDF (will use direct extraction)")
            return True
            
    except Exception as e:
        logger.warning(f"Could not analyze PDF type: {e}")
        return True  # Default to text extraction

def extract_text_with_ocr(pdf_bytes):
    """
    Extract text from scanned PDF using OCR (Tesseract)
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        str: Extracted text from all pages
    """
    if not OCR_AVAILABLE:
        raise Exception("OCR libraries not installed. Cannot process scanned PDFs.")
    
    try:
        logger.info("Starting OCR text extraction...")
        
        # Create a custom temp directory in the project folder (no admin needed)
        temp_dir = os.path.join(os.getcwd(), 'temp_ocr')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Set TESSDATA_PREFIX and TEMP environment variables for Tesseract
        old_temp = os.environ.get('TEMP')
        old_tmp = os.environ.get('TMP')
        old_tmpdir = os.environ.get('TMPDIR')
        
        # Point all temp variables to our custom folder
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
        os.environ['TMPDIR'] = temp_dir
        
        # Convert PDF to images with custom output folder
        images = convert_from_bytes(
            pdf_bytes, 
            dpi=300,
            output_folder=temp_dir,
            fmt='jpeg'
        )
        logger.info(f"Converted PDF to {len(images)} images (300 DPI)")
        
        text = ""
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)} with OCR...")
            
            # Perform OCR on the image with custom config
            custom_config = f'--tessdata-dir "{temp_dir}"'
            try:
                page_text = pytesseract.image_to_string(
                    image, 
                    lang='eng',
                    config=custom_config
                )
            except:
                # Fallback without custom config if it fails
                page_text = pytesseract.image_to_string(image, lang='eng')
            
            if page_text.strip():
                text += f"\n--- Page {page_num} ---\n{page_text}\n"
            else:
                logger.warning(f"Page {page_num} produced no OCR text")
        
        # Restore original environment variables
        if old_temp:
            os.environ['TEMP'] = old_temp
        if old_tmp:
            os.environ['TMP'] = old_tmp
        if old_tmpdir:
            os.environ['TMPDIR'] = old_tmpdir
        
        # Clean up temp files
        try:
            import shutil
            import glob
            # Remove image files
            for img_file in glob.glob(os.path.join(temp_dir, '*.jpg')):
                os.remove(img_file)
            for img_file in glob.glob(os.path.join(temp_dir, '*.jpeg')):
                os.remove(img_file)
            # Don't delete the folder itself, just clean it
        except Exception as e:
            logger.warning(f"Could not clean up temp files: {e}")
        
        logger.info(f"✓ OCR completed: Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"❌ OCR extraction failed: {e}")
        raise

def extract_text_from_pdf(pdf_bytes):
    """
    Intelligently extract text from PDF - uses direct extraction for text PDFs
    and OCR for scanned/image-based PDFs
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        logger.info(f"Processing PDF with {len(pdf_reader.pages)} pages")
        
        # Detect if PDF is text-based or scanned
        is_text_based = is_pdf_text_based(pdf_bytes)
        
        if is_text_based:
            # Use direct text extraction (faster)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                
                # Only add page text if it's meaningful (more than just whitespace)
                if page_text and len(page_text.strip()) > 10:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                else:
                    logger.warning(f"Page {page_num + 1} has little or no extractable text")
            
            logger.info(f"✓ Direct extraction: {len(text)} characters from PDF")
            
        else:
            # Use OCR for scanned PDFs
            text = extract_text_with_ocr(pdf_bytes)
        
        # Final validation
        if len(text.strip()) < 100:
            logger.warning(f"⚠️ Only extracted {len(text)} characters")
            logger.warning("PDF might be empty, encrypted, or have extraction issues")
        
        return text
        
    except Exception as e:
        logger.error(f"❌ Error extracting text from PDF: {e}")
        raise

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks with smart sentence boundaries
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        list: List of text chunks
    """
    # Split into words
    words = text.split()
    chunks = []
    
    # Create chunks with overlap
    i = 0
    while i < len(words):
        # Get chunk words
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text)
        
        # Move forward by (chunk_size - overlap)
        i += chunk_size - overlap
        
        # Break if we've processed all words
        if i >= len(words):
            break
    
    logger.info(f"✓ Created {len(chunks)} chunks from {len(words)} words")
    logger.info(f"   Chunk size: {chunk_size} words, Overlap: {overlap} words")
    
    return chunks

# ==================== EMBEDDINGS & RETRIEVAL ====================
def generate_embeddings(texts):
    """Generate embeddings for a list of texts"""
    try:
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        logger.info(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}")
        raise

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_chunks(query, top_k=TOP_K_RESULTS):
    """Retrieve the most relevant chunks for a query"""
    if not document_store["embeddings"]:
        return []
    
    query_embedding = embedding_model.encode([query])[0]
    
    similarities = []
    for idx, doc_embedding in enumerate(document_store["embeddings"]):
        sim_score = cosine_similarity(query_embedding, doc_embedding)
        similarities.append({
            "index": idx,
            "similarity": float(sim_score),
            "text": document_store["chunks"][idx],
            "metadata": document_store["metadata"][idx]
        })
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top_chunks = similarities[:top_k]
    
    logger.info(f"✓ Retrieved {len(top_chunks)} relevant chunks")
    for i, chunk in enumerate(top_chunks):
        logger.info(f"  Chunk {i+1}: Similarity = {chunk['similarity']:.4f}")
    
    return top_chunks

# ==================== ANSWER GENERATION ====================
def call_gemini_api(prompt, max_retries=3):
    """Call Gemini API with retry logic"""
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Gemini API (attempt {attempt + 1}/{max_retries})")
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    max_output_tokens=4096,
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            logger.warning(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All API attempts failed. Last error: {e}")
                raise

def generate_answer(question, context_chunks):
    """Generate answer using Gemini with retrieved context"""
    context = "\n\n".join([
        f"[Source {i+1}]: {chunk['text']}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

Context from the document:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context above
- If the context doesn't contain enough information, say "I don't have enough information in the documents to answer this question."
- Be concise and accurate
- If you use information from the sources, mention which source number (e.g., "According to Source 1...")

Answer:"""
    
    try:
        logger.info("Generating answer with Gemini...")
        answer = call_gemini_api(prompt)
        logger.info(f"✓ Generated answer ({len(answer)} characters)")
        return answer
        
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "PDF Chatbot API is running",
        "documents_loaded": len(document_store["chunks"]) > 0,
        "chunks_count": len(document_store["chunks"]),
        "models_initialized": embedding_model is not None and gemini_model is not None,
        "gemini_model": GEMINI_MODEL_NAME
    })

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a PDF document"""
    try:
        logger.info("\n" + "="*50)
        logger.info("📄 New document upload request")
        logger.info("="*50)
        
        if 'file' not in request.files:
            logger.warning("❌ No file in request")
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("❌ Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"❌ Invalid file type: {file.filename}")
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        logger.info(f"📁 Processing file: {file.filename}")
        
        pdf_content = file.read()
        logger.info(f"✓ Read {len(pdf_content)} bytes")
        
        text = extract_text_from_pdf(pdf_content)
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        
        document_store["chunks"] = chunks
        document_store["embeddings"] = embeddings.tolist()
        document_store["metadata"] = [
            {"filename": file.filename, "chunk_id": i}
            for i in range(len(chunks))
        ]
        
        logger.info("="*50)
        logger.info("✅ Document processed successfully!")
        logger.info(f"   Filename: {file.filename}")
        logger.info(f"   Total characters: {len(text):,}")
        logger.info(f"   Chunks created: {len(chunks)}")
        logger.info("="*50 + "\n")
        
        return jsonify({
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_characters": len(text)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Answer questions based on uploaded documents"""
    try:
        logger.info("\n" + "="*50)
        logger.info("💬 New chat request")
        logger.info("="*50)
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            logger.warning("❌ No question in request")
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        logger.info(f"❓ Question: {question}")
        
        if not document_store["chunks"]:
            logger.warning("❌ No documents loaded")
            return jsonify({
                "error": "No documents uploaded. Please upload a PDF document first."
            }), 400
        
        relevant_chunks = retrieve_relevant_chunks(question)
        answer = generate_answer(question, relevant_chunks)
        
        sources = [
            {
                "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "similarity": chunk["similarity"],
                "metadata": chunk["metadata"]
            }
            for chunk in relevant_chunks
        ]
        
        logger.info("="*50)
        logger.info("✅ Answer generated successfully!")
        logger.info("="*50 + "\n")
        
        return jsonify({
            "answer": answer,
            "sources": sources
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PDF CHATBOT - FLASK BACKEND")
    print("="*60)
    print("\n🚀 Starting application...\n")
    
    initialize_models()
    
    print("\n✅ Application ready!")
    print("="*60)
    print("Frontend: http://localhost:8000")
    print("API Endpoints:")
    print("  GET  /api/health   - Health check")
    print("  POST /api/upload   - Upload PDF document")
    print("  POST /api/chat     - Ask questions")
    print("="*60)
    print(f"Using Gemini Model: {GEMINI_MODEL_NAME}")
    print("="*60)
    print("\n🌐 Starting Flask server...\n")
    
    app.run(host="0.0.0.0", port=8000, debug=True)