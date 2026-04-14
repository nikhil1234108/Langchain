from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    output_dimensionality=32,
)

answers = [
    "The meaning of life is to create meaning through love, growth, and contribution.",
    "A woodchuck would chuck as much wood as determination and physics allow.",
    "The brain works through networks of neurons that send electrical and chemical signals to process thought, memory, and action.",
]

queries = [
    "What is the meaning of life?",
    "How much wood would a woodchuck chuck?",
    "How does the brain work?",
]

ans_embeddings = embeddings.embed_documents(answers)
query_embeddings = [embeddings.embed_query(q) for q in queries]

similarities = cosine_similarity(query_embeddings, ans_embeddings)

print(list(similarities))

for i, sim in enumerate(similarities):
    best_idx = np.argmax(sim)
    print(f"Query {queries[i]}")
    print(f"Best Answe:{answers[best_idx]}")
