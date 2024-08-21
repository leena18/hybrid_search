import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import nltk
import re

# Load environment variables
load_dotenv()
nltk.download('punkt')

# Streamlit UI styling
st.set_page_config(page_title="Pinecone Hybrid Search", page_icon="🔍", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #003366;
        color: white;
    }
    .sidebar .sidebar-content h2 {
        color: white;
    }
    .css-1v0mbdj {
        font-size: 24px;
        color: #1f77b4;
        text-align: center;
    }
    .css-1v0mbdj h1 {
        font-size: 36px;
    }
    .css-1v0mbdj h2 {
        font-size: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load API keys
api_key = os.getenv("PINECONE_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# Validate API keys
if not api_key or not hf_token:
    st.error("API keys are missing. Please check your .env file.")
    st.stop()

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)
index_name = "hybrid-search-langchain-pinecone"

# Create the Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    st.write("🔄 Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# Initialize embeddings and BM25 encoder
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()

# Initialize retriever
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index)

# Streamlit UI - Main Page
st.title("🔍 Pinecone Hybrid Search with Streamlit")
st.subheader("Explore hybrid search across dense and sparse embeddings")

# Streamlit Sidebar
st.sidebar.header("Add Texts")
texts_input = st.sidebar.text_area("Enter sentences (one per line):", value="In 2023, I visited Paris\nIn 2022, I visited New York\nIn 2021, I visited New Orleans")

if st.sidebar.button("Add Texts"):
    sentences = [s.strip() for s in texts_input.split("\n") if s.strip()]
    if sentences:
        retriever.add_texts(sentences)
        st.sidebar.success("Texts added to index!")
    else:
        st.sidebar.warning("No valid sentences provided. Please enter at least one sentence.")

st.sidebar.header("Query")
query = st.sidebar.text_input("Enter your query:", value="Which city did I visit first?")

# Function to extract the year from a sentence
def extract_year(sentence):
    match = re.search(r"\b(19|20)\d{2}\b", sentence)
    return int(match.group()) if match else None

# Search functionality
if st.sidebar.button("Search"):
    if query:
        with st.spinner("Searching..."):
            # Re-initialize the retriever to ensure it uses the latest index data
            results = retriever.invoke(query)

        st.write("Search Results:")

        # Sort the results by year
        results_with_years = []
        for res in results:
            year = extract_year(res.page_content)
            if year:
                results_with_years.append((year, res.page_content))

        # Determine whether the query is asking for the first or last visit
        if "first" in query.lower():
            results_with_years.sort(key=lambda x: x[0])  # Sort by earliest year
            selected_result = results_with_years[0][1] if results_with_years else None
        elif "last" in query.lower() or "latest" in query.lower():
            results_with_years.sort(key=lambda x: x[0], reverse=True)  # Sort by latest year
            selected_result = results_with_years[0][1] if results_with_years else None
        else:
            selected_result = None

        # Display the selected result
        if selected_result:
            st.markdown(f"**Answer:** {selected_result}")
        else:
            st.write("No relevant results found.")
    else:
        st.warning("Please enter a query to search.")
