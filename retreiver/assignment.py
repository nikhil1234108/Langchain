import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv



load_dotenv()

hf_token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("hugging_face_api_key")
)

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='conversational',
    temperature=0.4,
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(
    llm=llm
)
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
    documents=docs,
    embedding=HuggingFaceEndpointEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2',
        huggingfacehub_api_token=hf_token,
    )
)

multi_query = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={'k':2}),
    llm=model,
)

retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':3})
compressor = LLMChainExtractor.from_llm(model)
context_retriever = ContextualCompressionRetriever(
    base_retriever= retriever,
    base_compressor = compressor
)

results = multi_query.invoke('what james naismith use to do?')

result2 = context_retriever.invoke('what is photosynthasis?')

for i, doc in enumerate(results):
    print(f'{i+1}')
    print(doc.page_content)

for i, doc in enumerate(result2):
    print(f'{i+1}')
    print(doc.page_content)
