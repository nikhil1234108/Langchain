from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

class feedback(BaseModel):
    sentiment: Literal['positive', 'negitive'] = Field(description='Classify the sentiment of the review whether it is positive or negitive')

parser = PydanticOutputParser(pydantic_object=feedback)
parser2 = StrOutputParser()

template1 = PromptTemplate(
    template = "Classify the sentiment of the statement whether it is positive or negitive from the feedback {feedback} \n {format_instruction}.",
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':parser.get_format_instructions()},

)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text_generation",
    temperature=0.3,
)

model = ChatHuggingFace(llm=llm)
Classify_chain = template1 | model | parser

template2 = PromptTemplate(
    template = "write appropriate message to the positive feedback {feedback}.",
    input_variables = ['feedback'],
)
template3 = PromptTemplate(
    template = "write appropriate message for negitive feedback {feedback}.",
    input_variables = ['feedback'],
)
chain2 = template2 | model | parser2
chain3 = template3 | model | parser2
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive',chain2),
    (lambda x:x.sentiment == 'negitive',chain3),
    RunnableLambda(lambda x: "no sentiment founded")
 )

chain  = Classify_chain | branch_chain

result = chain.invoke({'feedback':'this is terrible restaurant'})
print(result)
chain.get_graph().print_ascii()