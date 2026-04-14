from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

path = Path(__file__).with_name('CV_Nikhil.pdf')

loader = PyPDFLoader(str(path))

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 7,
    separator = ''
)
recursive_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,
    chunk_size=70,
    chunk_overlap=15,
)
output = splitter.split_documents(docs)
output2 = recursive_splitter.split_documents(docs)


print(len(output))
print(output[4].page_content if output else "(no chunks)")
print(len(output2))
(print(output2[4].page_content))