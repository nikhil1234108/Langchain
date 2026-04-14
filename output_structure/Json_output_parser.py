from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text_generation",
    temperature=0.6,)
parser = JsonOutputParser()

model = ChatHuggingFace(llm=llm)
template = PromptTemplate(template="""
Give me the name, age, city of a fictional person \n {format_instruction}.""",
input_variables = [],
partial_variables={'format_instruction': parser.get_format_instructions()},)
prompt = template.format()
result = model.invoke(prompt)
print(result)
final_prompt = parser.parse(result.content)
print(final_prompt)

chain = template | model | parser
result2 = chain.invoke({})
print(result2)
