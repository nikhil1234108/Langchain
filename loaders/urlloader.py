from langchain_community.document_loaders import WebBaseLoader
import os

url = 'https://docs.langchain.com/oss/python/langchain/overview'
os.environ.setdefault("USER_AGENT", "LangChainLoader/1.0 (+https://docs.langchain.com)")
loader = WebBaseLoader(url)
docs=loader.load()
print(len(docs))
print(docs[0].page_content)
print(docs[0])