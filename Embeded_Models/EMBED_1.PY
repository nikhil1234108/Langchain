from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=32,
)

result = embedding.embed_query("Amaravathi is capital of Andhra Pradesh")
print(f"Vector length: {len(result)}")
print(result)
