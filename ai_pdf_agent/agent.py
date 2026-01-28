# ===============================
# agent.py
# ===============================

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


# ===============================
# BLOCK 1: LOAD PDF
# ===============================
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# ===============================
# BLOCK 1: SPLIT DOCUMENTS
# ===============================
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    return chunks


# ===============================
# BLOCK 2: CREATE EMBEDDINGS
# ===============================
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


# ===============================
# BLOCK 2: CREATE VECTOR STORE
# ===============================
def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ===============================
# BLOCK 2: BUILD RAG PIPELINE
# ===============================
def process_pdf(pdf_path):
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    embeddings = create_embeddings()
    vectorstore = create_vectorstore(chunks, embeddings)
    return vectorstore


# ===============================
# BLOCK 4: CREATE AGENT
# ===============================
def create_agent(vectorstore):
    # LLM (Ollama)
    llm = Ollama(
        model="llama3",   # or "llama3:8b" if you downloaded that
        temperature=0
    )

    # Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Retrieval QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # External Tool
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
            description="Perform math calculations"
        )
    ]

    # Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent
