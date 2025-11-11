import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from typing import TypedDict, List
from langgraph.graph import StateGraph
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

PDF_FOLDER = "pdf"
MEMORY_PATH = Path("./memory.json")
MEMORY_SIZE = 5  # include last N memory entries

# ------------------------- LOAD PDFs -------------------------
pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

text = ""
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    for page in pages:
        text += page.page_content

documents = [Document(page_content=text)]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=GROQ_API_KEY
)

def plan_query(query: str) -> str:
    decision = "retrieve" if len(query.split()) > 2 else "no_retrieve"
    print(f"[PLAN] Decision: {decision}")
    return decision

def retrieve_docs(query: str, k: int = 4) -> List[Document]:
    """Retrieve relevant documents."""
    results = retriever.invoke(query)
    print(f"[RETRIEVE] Retrieved {len(results)} docs")
    return results

def generate_answer(context: str, question: str) -> str:
    prompt = PromptTemplate(
        template=(
            "You are a strict context-aware assistant.\n"
            "Answer ONLY based on the provided context. "
            "If the context does not contain the answer, say: "
            "'The answer is not available in the provided context.'\n\n"
            "Question:\n{question}\n\n"
            "Context:\n{context}\n\nAnswer:"
        )
    )
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    print(f"[ANSWER] {response.content.strip()}")
    return response.content.strip()

def reflect_answer(question: str, answer: str) -> str:
    reflection_prompt = (
        "Evaluate the following response.\n\n"
        f"Question: {question}\nAnswer: {answer}\n\n"
        "Say 'Yes, relevant' or def'No, not relevant' and briefly explain why."
    )
    reflection = llm.invoke([HumanMessage(content=reflection_prompt)])
    print(f"[REFLECT] {reflection.content.strip()}")
    return reflection.content.strip()

class State(TypedDict):
    question: str
    plan: str
    context: List[Document]
    answer: str
    reflection: str

def plan(state: State) -> State:
    state["plan"] = plan_query(state["question"])
    return state

def retrieve(state: State) -> State:
    if state["plan"] == "retrieve":
        state["context"] = retrieve_docs(state["question"])
    else:
        state["context"] = []
    return state

def answer(state: State) -> State:
    context = " ".join(doc.page_content for doc in state["context"]) if state["context"] else ""
    state["answer"] = generate_answer(context, state["question"])
    return state

def reflect(state: State) -> State:
    state["reflection"] = reflect_answer(state["question"], state["answer"])
    return state

workflow = StateGraph(State)

workflow.add_node("plan", plan)
workflow.add_node("retrieve", retrieve)
workflow.add_node("answer", answer)
workflow.add_node("reflect", reflect)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "retrieve")
workflow.add_edge("retrieve", "answer")
workflow.add_edge("answer", "reflect")
workflow.set_finish_point("reflect")

graph = workflow.compile()

def run_agent(question: str):
    result = graph.invoke({"question": question})
    print("\n=== FINAL OUTPUT ===")
    print(f"Answer: {result['answer']}")
    print(f"Reflection: {result['reflection']}")
    return result

if __name__ == "__main__":
    run_agent("What is the main objective of the assignment?")
    run_agent("Who is the Prime Minister of India?")
