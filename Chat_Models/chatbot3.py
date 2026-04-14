from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

messages = [
    SystemMessage(content="What you want?"),
    HumanMessage(content="what are optimizers and loss function?")

]
result = model.invoke(messages)
messages.append(AIMessage(result.content))
print(messages)