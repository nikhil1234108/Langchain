from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation',
    temperature=0.6,
)

def wordcounter(text):
    return len(text.split())

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template="write a joke on the topic {topic}.",
    input_variables = ['topic']
)
template2 = PromptTemplate(
    template = "Explain the joke {joke}",
    input_variables=['joke']
)

template3 = PromptTemplate(
    template = "facts on the explanation {explanation}.",
    input_variables = ['explanation']
    )

chain1 = RunnableSequence(
    template1, model, parser,
    template2, model, parser
)

parallel_chain = RunnableParallel({
    'facts on joke':RunnablePassthrough(),
    'wordcount': RunnableLambda(wordcounter),
    })

final_chain = RunnableSequence(chain1,parallel_chain)
result=final_chain.invoke({'topic':'buffelo'})
print(result)


