from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import  HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

pdf_path = Path(__file__).with_name('CV_Nikhil.pdf')
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text_generation',
    temperature=0.6,
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()
template = PromptTemplate(
    template = "answer the questions {question} asked by the user from uploaded document {document}",
    input_variables = ['question', 'document']
)

chain = template | model | parser

result = chain.invoke({'document':docs[0].page_content,'question':'who is nikhil sai and what he wants to be?'})

print(docs)
print(len(docs))
print(result)