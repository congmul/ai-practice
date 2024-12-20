from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_core.vectorstores import InMemoryVectorStore

# Clear cached environment variables
os.environ.clear()
load_dotenv(override=True)

# Note that an individual Document object often represents a chunk of a larger document
# documents = [
#     Document(
#         page_content="Dogs are great companions, known for their loyalty and friendliness.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
#     Document(
#         page_content="Cats are independent pets that often enjoy their own space.",
#         metadata={"source": "mammal-pets-doc"},
#     ),
# ]

file_path = "./assets/documents/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))
# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

# The overlap helps mitigate the possibility of separating a statement from important context related to it
# We set add_start_index=True so that the character index where each split Document starts within the initial Document is preserved as metadata attribute “start_index”.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"len(all_splits): {len(all_splits)}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)
# print("===== ids =====>>")
# print(ids)
# print("===== ids =====<<")
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

results = vector_store.similarity_search_by_vector(embedding)
print(results[0])