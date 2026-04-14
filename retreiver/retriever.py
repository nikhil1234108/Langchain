from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError as exc:
        raise ImportError(
            "MultiQueryRetriever is not available in the installed LangChain packages. "
            "Use `langchain_classic.retrievers.multi_query` with LangChain 1.x, "
            "or install a version that exposes `langchain.retrievers.multi_query`."
        ) from exc

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-001'
)

vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding
)

multi_query = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    llm=ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0.6
    )
)

query = "how to improve energy levels and maintain balance?"
results = multi_query.invoke(query)

for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(doc.page_content)
