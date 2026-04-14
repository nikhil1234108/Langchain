import sys

from langchain_community.retrievers import WikipediaRetriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

retriever = WikipediaRetriever(top_k_results=2, lang="en")
query = "History of south india"

docs=retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\nResult>> {i+1}.....")
    print(f"doc:\n{doc.page_content}....")


