from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from pathlib import Path

path = Path(__file__).with_name('Director1')

loader = DirectoryLoader(
    path=path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    silent_errors=True,
)
docs = loader.load()
print(len(docs))
print(docs[0].page_content)
print(f"meta_content{docs[0].metadata}")

docs1 = loader.lazy_load()
for doc in docs1:
    print(doc.metadata)
