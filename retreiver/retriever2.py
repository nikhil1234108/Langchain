from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Step 1: Your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models.")
]

vectorstore = Chroma.from_documents(
    documents = documents,
    embedding= GoogleGenerativeAIEmbeddings(
        model = 'gemini-embedding-001',
        collection_name = 'my_collection'
    )
)
retriever = vectorstore.as_retriever(search_type="mmr",
search_kwargs={"k":2,"lambda_mult":0})

query = "what is chroma used for?"

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n..{i+1}")
    print(doc.page_content)
