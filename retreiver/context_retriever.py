from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from langchain_classic.retrievers.contextual_compression import (
        ContextualCompressionRetriever,
    )
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    try:
        from langchain.retrievers.contextual_compression import (
            ContextualCompressionRetriever,
        )
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError as exc:
        raise ImportError(
            "Contextual compression retrievers are not available in the installed "
            "LangChain packages. Install `langchain-classic` for LangChain v1, or "
            "use an older LangChain version that still exposes `langchain.retrievers`."
        ) from exc

load_dotenv()

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

vectorstore = FAISS.from_documents(
    documents = docs,
    embedding = GoogleGenerativeAIEmbeddings(
        model = 'gemini-embedding-001'
    )
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash',
    temperature = 0.6
)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_retriever = retriever,
    base_compressor = compressor
)
# Query the retriever
query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(i+1)
    print(doc)
