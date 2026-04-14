from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

docs = [
    Document(
        page_content=(
            "Bharatanatyam is a major form of Indian classical dance from Tamil Nadu. "
            "It emphasizes precise footwork, expressive hand gestures (mudras), and storytelling from Hindu epics. "
            "Traditionally performed in temples; today also staged in theatres worldwide."
        ),
        metadata={"form": "Bharatanatyam", "abbr": "BT", "state": "Tamil Nadu"},
    ),
    Document(
        page_content=(
            "Kathak originated in North India and blends storytelling with rhythmic footwork and spins. "
            "It developed in both Hindu temples and Mughal courts, giving it intricate tatkar patterns "
            "and graceful chakkars (pirouettes)."
        ),
        metadata={"form": "Kathak", "abbr": "KT", "state": "Uttar Pradesh"},
    ),
    Document(
        page_content=(
            "Kathakali is a highly stylised dance-drama from Kerala, known for elaborate costumes, "
            "face paint, and exaggerated expressions. Performances often depict stories from the Mahabharata and Ramayana."
        ),
        metadata={"form": "Kathakali", "abbr": "KK", "state": "Kerala"},
    ),
    Document(
        page_content=(
            "Odissi comes from Odisha and is characterised by tribhangi posture, fluid torso movements, "
            "and sculpturesque poses inspired by temple reliefs. It often celebrates devotion to Lord Jagannath."
        ),
        metadata={"form": "Odissi", "abbr": "OD", "state": "Odisha"},
    ),
    Document(
        page_content=(
            "Manipuri dance from Manipur is gentle and lyrical, often performed as part of Ras Lila depicting Krishna’s life. "
            "Costumes include the cylindrical skirt and light, flowing movements distinct from other classical styles."
        ),
        metadata={"form": "Manipuri", "abbr": "MP", "state": "Manipur"},
    ),
    Document(
        page_content=(
            "Kuchipudi is a classical dance form from Andhra Pradesh, named after the village of Kuchipudi. "
            "It combines fast footwork and expressive abhinaya with dramatic episodes and sometimes speech. "
            "Dancers may perform on the rim of a brass plate (tarangam) while balancing a pot of water, showcasing agility and control."
        ),
        metadata={"form": "Kuchipudi", "abbr": "KC", "state": "Andhra Pradesh"},
    ),
    Document(
        page_content=(
            "Perini Shivatandavam (often called Perini Shiva Natyam) is a vigorous classical dance from Telangana, "
            "rooted in the martial and devotional traditions of the Kakatiya era. Male dancers perform powerful, "
            "leaping steps and rhythmic footwork in honour of Lord Shiva, evoking the spirit of warriors before battle. "
            "It was reconstructed in the 20th century from historical sources and temple sculpture."
        ),
        metadata={"form": "Perini Shivatandavam", "abbr": "PS", "state": "Telangana"},
    ),
]

persist_dir = Path(__file__).resolve().parent / "my_chroma_db_dance"

vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=32,
    ),
    persist_directory=str(persist_dir),
    collection_name="classical_dance",
)

vectorstore.add_documents(docs)

print(vectorstore.get(include=["embeddings", "documents", "metadatas"]))
print(
    vectorstore.similarity_search_with_score(
        query="which classical dance is from Tamil Nadu?",
        k=1,
    )
)

doc2 = Document(
        page_content=(
            "Kuchipudi is a classical dance form from Andhra Pradesh, named after the village of Kuchipudi. "
            "It combines fast footwork and expressive abhinaya with dramatic episodes and sometimes speech. "
            "Dancers may perform on the rim of a brass plate (tarangam) while balancing a pot of water, showcasing agility and control."
        ),
        metadata={"form": "Kuchipudi", "abbr": "KC", "state": "Andhra Pradesh"},
    )
vectorstore.update_documents(
    ids=["f0215ced-12b1-441f-8eae-182382a3baf0"],
    documents=[doc2],
)
print(vectorstore.get(include=['embeddings','documents','metadatas']))


