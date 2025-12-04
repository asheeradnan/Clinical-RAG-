#to convert your chunked data to mylangchain index

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import json
import numpy as np


# Load your chunks and embeddings from jsonl
text_embeddings = []

with open("chunked_docs_embeddings.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        text_embeddings.append((item['text'], np.array(item['embedding'], dtype="float32")))

# Initialize embeddings object (needed by LangChain)
hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build FAISS index with precomputed embeddings
faiss_db = FAISS.from_embeddings(text_embeddings, hf_embeddings)

# Save to disk
faiss_db.save_local("langchain_index")

# Query example
results = faiss_db.similarity_search("Why am I diabetic?", k=5)
print([r.page_content for r in results])