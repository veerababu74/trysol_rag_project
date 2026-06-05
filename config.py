
import os
import tempfile
from dotenv import load_dotenv

# load_dotenv() reads the .env file and puts the values into the environment.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# "openai" or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

VECTOR_STORE = "faiss"  # only FAISS is supported

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mistral-embed")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

_TMP = tempfile.gettempdir()

UPLOAD_DIR = os.path.join(_TMP, "uploaded_files")                   # where uploaded files are saved
FAISS_DIR = os.path.join(_TMP, "vector_stores", "faiss_index")  # where the FAISS index is saved

# Chunking defaults
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
