from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7,
    
)
chat = ChatHuggingFace(llm=llm)

response = chat.invoke("Who is Albert Einstein?")

print(response.content)