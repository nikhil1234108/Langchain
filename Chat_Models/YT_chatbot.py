import sys
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

load_dotenv()

video_id = "J5_-l7WIO_w"

transcript = ""
transcript_list = []

try:
    print(f"Fetching transcript for video ID: {video_id}...")
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi", "en"])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    print("Transcript fetched successfully!")
    
except TranscriptsDisabled:
    print("Error: Transcripts are disabled for this video.")
    sys.exit(1)
except NoTranscriptFound:
    print("Error: No transcript found in the requested languages ('hi', 'en').")
    sys.exit(1)
except VideoUnavailable:
    print("Error: Video is unavailable.")
    sys.exit(1)
except ET.ParseError:
    print("Error: YouTube returned an empty XML response (ParseError).")
    print("Fix: You are likely rate-limited by YouTube or need to run: pip install --upgrade youtube-transcript-api")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred while fetching the transcript: {e}")
    sys.exit(1)

if not transcript:
    print("Error: Transcript is empty. Stopping execution.")
    sys.exit(1)

print("Processing transcript and building vector store...")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})



llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.6,
    max_retries=2
)
multi_query = MultiQueryRetriever.from_llm(
    base_retriever=retriever,
    llm=llm
)


prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
    input_variables=["context", "question"]
)

def retriever_format(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    "context": multi_query | RunnableLambda(retriever_format),
    "question": RunnablePassthrough()
})

parser = StrOutputParser()

chain = parallel_chain | prompt | llm | parser

print("Invoking chain...\n")
print("-" * 50)
result = chain.invoke("What are the key points of the video?")
print(result)
print("-" * 50)