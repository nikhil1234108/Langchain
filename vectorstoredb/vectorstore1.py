from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

docs = [
    Document(
        page_content=(
            "Chennai Super Kings (CSK) is based in Chennai, Tamil Nadu. "
            "Known for consistent playoffs appearances and MS Dhoni's leadership. "
            "Home ground: M. A. Chidambaram Stadium."
        ),
        metadata={"team": "Chennai Super Kings", "abbr": "CSK", "city": "Chennai"},
    ),
    Document(
        page_content=(
            "Mumbai Indians (MI) is based in Mumbai, Maharashtra. "
            "The most successful IPL franchise by titles. "
            "Home ground: Wankhede Stadium."
        ),
        metadata={"team": "Mumbai Indians", "abbr": "MI", "city": "Mumbai"},
    ),
    Document(
        page_content=(
            "Royal Challengers Bengaluru (RCB) is based in Bengaluru, Karnataka. "
            "Known for star batters and passionate home crowds. "
            "Home ground: M. Chinnaswamy Stadium."
        ),
        metadata={"team": "Royal Challengers Bengaluru", "abbr": "RCB", "city": "Bengaluru"},
    ),
    Document(
        page_content=(
            "Kolkata Knight Riders (KKR) is based in Kolkata, West Bengal. "
            "Two-time IPL champions with a strong spin legacy. "
            "Home ground: Eden Gardens."
        ),
        metadata={"team": "Kolkata Knight Riders", "abbr": "KKR", "city": "Kolkata"},
    ),
    Document(
        page_content=(
            "Rajasthan Royals (RR) is based in Jaipur, Rajasthan. "
            "Inaugural IPL champions; known for scouting young talent. "
            "Home ground: Sawai Mansingh Stadium."
        ),
        metadata={"team": "Rajasthan Royals", "abbr": "RR", "city": "Jaipur"},
    ),
]

persist_dir = Path(__file__).resolve().parent / "my_chroma_db"

vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=32,
    ),
    persist_directory=str(persist_dir),
    collection_name="simple",
)

vectorstore.add_documents(docs)

print(vectorstore.get(include=["embeddings", "documents", "metadatas"]))
print(vectorstore.similarity_search_with_score(
    query = 'which team plays in chinnaswamy stadium?',
    k=1
))
