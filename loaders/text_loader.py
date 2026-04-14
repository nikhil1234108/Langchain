from pathlib import Path

from langchain_community.document_loaders import TextLoader

cricket_path = Path(__file__).with_name("cricket.txt")
loader = TextLoader(str(cricket_path), encoding="utf-8")

docs = loader.load()
print(type(docs))
print(docs[0])
