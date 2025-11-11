import streamlit as st
from main import graph 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

st.set_page_config(page_title="Context Chatbot", layout="wide")
st.title("🤖 Context-Aware Chatbot")
 
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None
 
st.subheader("Upload PDF to add Context")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded:
    with st.spinner("Processing PDF..."):
        os.makedirs("temp_pdf", exist_ok=True)
        file_path = os.path.join("temp_pdf", uploaded.name)
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        full_text = "\n".join([p.page_content for p in pages])
        documents = [Document(page_content=full_text)]

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(docs, embeddings)
        st.session_state.retriever = vector_db.as_retriever()

    st.success("Context Updated! You can start chatting.")

 
st.markdown("### 💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


prompt = st.chat_input("Ask something based on the uploaded PDF...")

if prompt:
    if st.session_state.retriever is None:
        st.error("Please upload a PDF first.")
    else: 
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
 
        with st.spinner("Thinking..."):
            result = graph.invoke({"question": prompt})

        final_answer = result.get("answer", "Error: No answer returned.")
        plan_step = result.get("plan", "")
        retrieved_docs = result.get("context", [])
        reflection = result.get("reflection", "")
 
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        with st.chat_message("assistant"):
            st.write(final_answer)
 
            with st.expander("Show Internal Reasoning Steps"):
                st.markdown(f"**Plan:** `{plan_step}`")
                st.markdown(f"**Retrieved Documents:** {len(retrieved_docs)} found")
                st.markdown("**Generated Answer:**")
                st.write(final_answer)
                st.markdown("**Reflection:**")
                st.write(reflection)
