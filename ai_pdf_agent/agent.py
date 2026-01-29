# ===============================
# agent.py
# ===============================

# ============================================================
# AUTO-INSTALL MISSING DEPENDENCIES
# ============================================================
import sys
import subprocess
import importlib

REQUIRED_PACKAGES = {
    "langchain": "langchain",
    "langchain_community": "langchain-community",
    "langchain_huggingface": "langchain-huggingface",
    "faiss": "faiss-cpu",
    "sentence_transformers": "sentence-transformers",
    "pypdf": "pypdf",
    "ollama": "ollama",
}

def install_if_missing(import_name, pip_name):
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 Installing missing package: {pip_name}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name]
        )

for import_name, pip_name in REQUIRED_PACKAGES.items():
    install_if_missing(import_name, pip_name)

print("✅ All required libraries are installed.\n")


# ============================================================
# IMPORTS (AFTER CHECKING LIBRARIES)
# ============================================================

# ----------- LLM -----------
from langchain_community.llms import Ollama

# ----------- PDF Loading -----------
from langchain.document_loaders import PyPDFLoader

# ----------- Text Splitting -----------
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------- Embeddings + Vector DB -----------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ----------- RAG + Agent -----------
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import Calculator


# ============================================================
# BLOCK 1: LOAD PDF
# ============================================================
def load_pdf(pdf_path):
    """
    Loads a PDF file and returns LangChain documents
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# ============================================================
# BLOCK 2: SPLIT DOCUMENTS
# ============================================================
def split_documents(documents):
    """
    Splits documents into overlapping text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ============================================================
# BLOCK 3: CREATE EMBEDDINGS
# ============================================================
def create_embeddings():
    """
    Creates HuggingFace sentence embeddings
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# ============================================================
# BLOCK 4: CREATE VECTOR STORE
# ============================================================
def create_vectorstore(chunks, embeddings):
    """
    Stores embeddings in FAISS vector database
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ============================================================
# BLOCK 5: PROCESS PDF (RAG PIPELINE)
# ============================================================
def process_pdf(pdf_path):
    """
    Complete PDF → Chunks → Embeddings → VectorStore pipeline
    """
    print("📄 Loading PDF...")
    documents = load_pdf(pdf_path)

    print("✂️ Splitting document into chunks...")
    chunks = split_documents(documents)

    print("🧠 Creating embeddings...")
    embeddings = create_embeddings()

    print("📦 Creating FAISS vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)

    print("✅ PDF processing completed.\n")
    return vectorstore


# ============================================================
# BLOCK 6: CREATE AGENT
# ============================================================
def create_agent(vectorstore):
    """
    Creates a conversational RAG-based agent with tools
    """

    # LLM (Ollama)
    llm = Ollama(
        model="llama3",   # or "llama3:8b"
        temperature=0
    )

    # Conversation Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # External Tool (Calculator)
    calculator = Calculator()

    tools = [
        Tool(
            name="Document QA",
            func=qa_chain.run,
            description="Answer questions from the uploaded PDF"
        ),
        Tool(
            name="Calculator",
            func=calculator.run,
            description="Perform mathematical calculations"
        )
    ]

    # Agent Initialization
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    print("🤖 Agent successfully created.\n")
    return agent


# ============================================================
# OPTIONAL: MAIN EXECUTION (CLI TEST)
# ============================================================
if __name__ == "__main__":
    pdf_path = "sample.pdf"   # change this to your PDF path

    vectorstore = process_pdf(pdf_path)
    agent = create_agent(vectorstore)

    print("💬 Ask questions about your PDF (type 'exit' to quit)\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = agent.run(query)
        print(f"Agent: {response}\n")
