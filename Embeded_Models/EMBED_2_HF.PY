from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",task="feature-extraction")
result = embeddings.embed_query("Amaravathi is capital of Andhra Pradesh")
print(f"Vector length: {len(result)}")
print(result)
