import google.generativeai as genai
import chromadb
from .config import settings

# Configure the generative AI model
genai.configure(api_key=settings.google_ai_api_key)

# Set up models
embedding_model = genai.get_base_model('models/text-embedding-004')
generation_model = genai.GenerativeModel('gemini-1.5-flash')

# Set up an in-memory vector database client
client = chromadb.PersistentClient(path="./chroma_db")
# Create a collection (or get it if it already exists)
# A collection is like a table in a traditional database.
collection = client.get_or_create_collection(
    name="context_aware_collection"
)