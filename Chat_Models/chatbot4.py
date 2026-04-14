from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "what can i do for you?"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", '{query}')
])
chat_history =[]
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

print(chat_history)

result = chat_template.invoke({
    "chat_history":chat_history,
    "query":"how many seasons are there in the game of thrones?"
})
print(result)
