from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """You’ve got two separate issues:

PyPDFLoader doesn’t take encoding=, so that argument must be removed.
The PDF you’re loading looks not actually a valid PDF (invalid pdf header: b'a%PDF'), which usually means the file is corrupted or has extra bytes at the beginning.
I’ll open loaders/pdf_loader.py, find which PDF it’s pointing at, then update the script to (a) call PyPDFLoader correctly and (b) sanity-check the PDF header with a clear error message.

Great, we can reproduce the problem: LOR.pdf starts with an extra byte (b'a%PDF...') so pypdf rejects it.

Next I’ll update pdf_loader.py to automatically detect this, create a repaired copy (by trimming everything before the first %PDF), and then load the repaired PDF.

Checking what your script prints so we can explain the numbers.

Here’s what those numbers mean in your script and how length works in the splitters.

What 67 and 63 are
They are not “length of one document” in characters.

len(output) → 67
split_documents(docs) returns a new list of chunk-Documents.
len(output) is how many chunks the CharacterTextSplitter produced (over all pages combined).

len(output2) → 63
Same idea for recursive_splitter: number of chunks after recursive splitting.

So: len(...) here is Python’s list length = count of items, not PDF page count and not “size of one doc” unless you pass a single string.

How the splitter decides chunk size
For both splitters, chunk_size and chunk_overlap apply to the text string being split (each page’s page_content, then merged across splits).

By default, “length” = character count:

The splitter uses something equivalent to len(text) on each piece (default length function is character-based).
So chunk_size=50 means it tries to build chunks of about 50 characters (not 50 words).
With separator='' on CharacterTextSplitter, it splits per character, so you get hard cuts in the middle of words — e.g. logy is the tail of Technology after a 50-char boundary.
If you meant “how many PDF pages” (docs)
len(docs) = number of Document objects from loader.load() — for a PDF, that’s usually one Document per page.
Summary:

len(docs) → number of loaded pages (documents from the loader).
len(output) / len(output2) → number of chunks after splitting.
Chunk boundaries → driven by chunk_size/chunk_overlap using character length by default, which is why you see broken words like logy.
"""

splitter = CharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap=5,
    separator = '|'
)
text_splitter = SemanticChunker(
    GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type = "standard_deviation",
    breakpoint_threshold_amount = 1
)
result = splitter.split_text(text)

result2 = text_splitter.split_text(text)
print(result)
print(result2)