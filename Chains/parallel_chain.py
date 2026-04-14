from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id='meta-llama/Llama-3.1-8B-Instruct',
task='text_generation',
temperature=0.6,)
model1 = ChatHuggingFace(llm=llm)
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash',
temperature=0.6,
max_retries = 2,
timeout=None,
max_tokens=None,
)
template1 = PromptTemplate(
    template = 'generate breif and detailed report on topic {topic}',
    input_variables = ['topic'],
)
template2 = PromptTemplate(
    template = 'generate a short and simple notes from the following text \n{text}.',
    input_variables = ['text'],
)

template3 = PromptTemplate(
    template = 'generate a Multiple choice questions(MCQs) from the text \n{text}',
    input_variables = ['text'],
)
template4 = PromptTemplate(
    template = 'merge the provided notes and MCQS into a single document \n notes {notes}, MCQs {MCQs}.',
    input_variables = ['notes','MCQs']
)

parser = StrOutputParser()

topic_generator = template1 | model1 | parser

parallel_processer = RunnableParallel({
    'notes': topic_generator | template2 | model2 | parser,
    'MCQs' : topic_generator | template3 | model2 | parser
})

merge_chain = template4 | model1 | parser
chain = parallel_processer | merge_chain

result=chain.invoke({'topic':'Bahubali'})
print(result)
chain.get_graph().print_ascii()
