from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation',
    temperature=0.6,
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Step 1 → facts
template1 = PromptTemplate(
    template="give interesting facts about the topic {topic}",
    input_variables=['topic']
)

# Step 2 → keywords
template2 = PromptTemplate(
    template="extract keywords from: {facts}",
    input_variables=['facts']
)

# Step 3 → places
template3 = PromptTemplate(
    template="extract place names from: {keywords}",
    input_variables=['keywords']
)

# Chain → topic → facts → keywords
main_chain = RunnableSequence(
    template1, model, parser,
    template2, model, parser
)

# 🔥 Use Passthrough here
final_chain = main_chain | RunnableParallel({
    "keywords": RunnablePassthrough(),  # passes keywords directly
    "places_name": template3 | model | parser  # uses same keywords
})

result = final_chain.invoke({'topic': 'the last nizam of hyderabad'})
print(result)