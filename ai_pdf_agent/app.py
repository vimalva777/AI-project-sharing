# ===============================
# app.py
# ===============================

import streamlit as st
import tempfile
from agent import process_pdf, create_agent

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AI PDF Agent (Ollama)",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Agent (FREE • Local • Ollama)")
st.write("Upload a PDF and chat with it using a local AI agent.")

# -------------------------------
# Session State
# -------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# PDF Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF and building knowledge base..."):
        try:
            vectorstore = process_pdf(pdf_path)
            st.session_state.agent = create_agent(vectorstore)
            st.success("✅ PDF processed successfully! You can start chatting.")
        except Exception as e:
            st.error(f"❌ Failed to process PDF: {e}")

# -------------------------------
# Chat Input
# -------------------------------
query = st.text_input(
    "Ask a question about the document",
    placeholder="e.g. Summarize the document, explain regression, calculate total cost..."
)

# -------------------------------
# Run Agent
# -------------------------------
if query and st.session_state.agent:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.agent.run(query)

            # Save chat history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", response))

        except Exception as e:
            st.error(f"❌ Error while generating response: {e}")

# -------------------------------
# Display Chat History
# -------------------------------
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 AI:** {message}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with LangChain + Ollama + FAISS • 100% Local & Free")
