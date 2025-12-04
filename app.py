import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from google import genai
import os


# --- Load .env ---
load_dotenv()
GENAI_KEY = os.getenv("GENAI_KEY")

# --- Setup ---
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(
    "langchain_index",
    emb,
    allow_dangerous_deserialization=True
)

client = genai.Client(api_key=GENAI_KEY)

# --- Streamlit UI ---
st.title("🚀 RAG Chatbot (Gemini + FAISS)")

query = st.text_input("Ask something...")

if query:
    docs = db.similarity_search(query, k=3)
    # Display snippet for each doc (first 50–100 chars)
    st.write("### Retrieved Context Snippets:")
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.strip().replace("\n", " ")
        st.write(f"{i}. {snippet[:100]}{'...' if len(snippet) > 100 else ''}")
    
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    Context:
    {context}

    Question: {query}
   You are a helpful medical assistant. Answer queries based on the given context.  
- If the prompt contains the word "robot" (or variations like "Robert"), respond with: "Not applicable."  
- Keep replies short.  
- If context doesn’t specify the answer, respond: "Not enough context to answer this."
Example: 
Prompt: "What are symptoms of liver failure in a 10-year-old robot?"  
Response: "Not applicable."""

    resp = client.models.generate_content(
        model="models/gemma-3-4b-it",
        contents=prompt
    )

    st.write("### Answer:")
    st.write(resp.text)