# Generative AI Agent using LangGraph & RAG

A **context-aware Question Answering AI Agent** built with **LangGraph**, **LangChain**, and **Groq LLaMA**, capable of answering questions **strictly from uploaded PDFs** using **Retrieval-Augmented Generation (RAG)**.  

This project was created as part of the **Generative AI Engineer Internship Task** for **SwarmLens / Aatoon Solutions**.

---

## Features

**LangGraph Agent Workflow** with 4 nodes:
- **Plan:** Decides if retrieval is required  
- **Retrieve:** Fetches context using FAISS embeddings  
- **Answer:** Generates context-grounded response  
- **Reflect:** Evaluates answer relevance  

**RAG Pipeline** with HuggingFace embeddings  
**Groq LLaMA model** for high-speed inference  
**Streamlit Chat UI** (WhatsApp-style)  
**Context-Only Answers** — no hallucinations  
**PDF Upload Support**  
**Detailed internal reasoning logs (plan → retrieve → answer → reflect)**  

---

## Installation Guide

### Clone the repository

```bash
git clone https://github.com/princ0301/Langgraph-Q-A-Chatbot.git
cd Langgraph-Q-A-Chatbot
```

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the Project

### Run the LangGraph Agent (CLI)

```bash
python main.py
```

### Run the Streamlit Chat UI

```bash
streamlit run app.py
```

Then open the app in your browser (default: http://localhost:8501).

---
