import streamlit as st 
import faiss
import numpy as np
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

#CONFIGURATION
API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=API_KEY)
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

vector_dim = 384  # Match embedding output size

#Initialize FAISS Index in Session State
if "faiss_index_layer1" not in st.session_state:
    st.session_state.faiss_index_layer1 = faiss.IndexFlatL2(vector_dim)  # Keywords/Summaries
    st.session_state.faiss_index_layer2 = faiss.IndexFlatL2(vector_dim)  # Full Documents
    st.session_state.doc_texts = {}  # Store full document texts
    st.session_state.summary_texts = {}  # Store summary texts

#FUNCTION: Extract text from PDFs
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF using PDFPlumber"""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

#FUNCTION: Call Groq API for Summaries/Keywords
def query_groq(text, prompt):
    """Calls Groq API for summarization or keyword extraction using SDK"""
    try:
        completion = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL"),
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text[:4096]}],
            temperature=0.7,
            max_tokens=10000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

#FUNCTION: Store Embeddings in FAISS
def store_embeddings(text, doc_id, index, storage):
    """Stores document embeddings in FAISS"""
    embedding = np.array(EMBEDDING_MODEL.encode(text)).reshape(1, -1)
    index.add(embedding)
    storage[doc_id] = text  # Store the actual text for retrieval

#FUNCTION: Retrieve Documents from FAISS
def retrieve_documents(query, index, storage):
    """Retrieves most relevant documents using FAISS"""
    if index.ntotal == 0:
        return None  # No documents available

    query_embedding = np.array(EMBEDDING_MODEL.encode(query)).reshape(1, -1)
    _, indices = index.search(query_embedding, k=3)

    if len(indices) == 0 or len(indices[0]) == 0 or indices[0][0] == -1:
        return None  # No relevant documents found

    return [storage[list(storage.keys())[i]] for i in indices[0] if i < len(storage)]


#FUNCTION: Guardrails for Response Filtering
def enforce_guardrails(query, response):
    """Applies guardrails to prevent hallucinations and restrict tax-related responses."""
    
    # If response is a list, join it into a single string for checking
    if isinstance(response, list):
        response_text = " ".join(response)
    else:
        response_text = response
    
    if "error" in response_text.lower() or "not found" in response_text.lower():
        return "⚠️ Sorry, I couldn't find a reliable answer."

    if "tax" in query.lower() and "tax" not in response_text.lower():
        return "⚠️ This system is restricted from providing non-tax-related answers."

    return response


#DECISION-MAKING AGENT
def decision_making_agent(query):
    """Retrieves answer from documents and LLM separately, displaying both in UI."""

    # 1️⃣ Try RAG (FAISS Retrieval)
    doc_based_answer = "⚠️ No relevant documents found."
    results_layer1 = retrieve_documents(query, st.session_state.faiss_index_layer1, st.session_state.summary_texts)

    if results_layer1:
        results_layer2 = retrieve_documents(results_layer1[0], st.session_state.faiss_index_layer2, st.session_state.doc_texts)
        retrieved_text = results_layer2 if results_layer2 else results_layer1
        
        # Ensure retrieved text is not empty before querying LLM
        if retrieved_text and len(retrieved_text[0]) > 10:  # Avoid empty or very short responses
            doc_based_answer = query_groq("\n".join(retrieved_text), 
                                          f"Answer the following query based only on the provided document context:\n{query}")

    # 2️⃣ Get an answer from LLM (Query Only)
    llm_based_answer = query_groq(query, "Provide a well-structured response based on general knowledge and reasoning:")

    return {
        "📚 Document-Based Answer": doc_based_answer,
        "🤖 LLM-Based Answer": llm_based_answer
    }



st.set_page_config(page_title="Decision-Making Agent with H-RAG", layout="wide")
st.title("🤖 Decision-Making Agent with Hierarchical RAG")

# Upload PDFs
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    st.sidebar.success("Files Uploaded!")

# Process Uploaded PDFs
if st.sidebar.button("Process Documents"):
    for file in uploaded_files:
        with st.spinner(f"Processing {file.name}..."):
            text = extract_text_from_pdf(file)
            summary = query_groq(text, "Summarize the following tax-related document in 150 words:")
            keywords = query_groq(text, "Extract the top 5 most important keywords related to tax from the following text:")

            # Store embeddings in session_state FAISS index
            store_embeddings(summary, file.name, st.session_state.faiss_index_layer1, st.session_state.summary_texts)
            store_embeddings(text, file.name, st.session_state.faiss_index_layer2, st.session_state.doc_texts)

            st.write(f"✅ **{file.name} Processed**")
            st.write(f"📄 **Summary:** {summary}")
            st.write(f"🔑 **Keywords:** {keywords}")

# Query Section
st.header("🔍 Ask a Question")
user_query = st.text_input("Enter your query:")

if st.button("Search"):
    with st.spinner("Generating answers..."):
        results = decision_making_agent(user_query)

    # Ensure UI updates correctly
    if results:
        # Display document-based answer
        st.subheader("📚 Document-Based Answer")
        st.write(results.get("📚 Document-Based Answer", "⚠️ No answer generated."))

        # Display LLM-based answer
        st.subheader("🤖 LLM-Based Answer")
        st.write(results.get("🤖 LLM-Based Answer", "⚠️ No answer generated."))

