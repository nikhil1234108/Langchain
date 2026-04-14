from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
# Specify the model explicitly (otherwise LangChain raises a validation error).
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
chat_history = [SystemMessage(content="what can I do for you?")]
while True:
    user_input = input("user:")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)
