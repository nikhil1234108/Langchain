from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash',
temperature=0.7,
max_tokens=None,
timeout=None,
max_retries=2,
)

result=llm.invoke("what is the capital of AndhraPradesh in 2026")
print(result.content)


