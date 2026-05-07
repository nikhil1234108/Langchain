from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field


class FactsSchema(BaseModel):
    fact_1: str = Field(description="fact_1 about the topic")
    fact_2: str = Field(description="fact_2 about the topic")
    fact_3: str = Field(description="fact_3 about the topic")


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    temperature=1,
)

model = ChatHuggingFace(llm=llm)
parser = PydanticOutputParser(pydantic_object=FactsSchema)
template = PromptTemplate(
    template=(
        "Give 3 facts about the topic {topic}.\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "mr.osho"})

print(result.model_dump_json(indent=2))
