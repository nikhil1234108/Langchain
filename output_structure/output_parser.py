from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)
template1 = PromptTemplate.from_template("""
write a detailed report in {topic}.
topic  :{topic}
""")
template2 = PromptTemplate.from_template("""
write 5 line summary on text \n{text}
text:\n{text}""")

prompt1 = template1.invoke({"topic": "Black Hole"})
result1 = model.invoke(prompt1)
print(result1.content)

prompt2 = template2.invoke({"text": result1.content})
result2 = model.invoke(prompt2)
print(result2.content)
