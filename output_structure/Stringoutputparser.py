from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    temperature=0.6,
)
model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate.from_template(
    """
write a detailed report on {topic}.
topic: {topic}
""".strip()
)

template2 = PromptTemplate.from_template(
    """
write a 5 line summary about:
{text}
""".strip()
)

parser = StrOutputParser()

chain = (
    template1
    | model
    | parser
    | RunnableLambda(lambda text: {"text": text})
    | template2
    | model
    | parser
)

result = chain.invoke({"topic": "black hole"})
print(result)
