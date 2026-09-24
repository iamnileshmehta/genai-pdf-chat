import os

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

load_dotenv()

# Free Groq model IDs (verify at https://console.groq.com/docs/models):
#   "llama-3.1-8b-instant"     -> fastest, most generous free limits
#   "llama-3.3-70b-versatile"  -> better answers, tighter free limits
GROQ_MODEL = "llama-3.1-8b-instant"


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


groq_api_key = get_api_key()
if not groq_api_key:
    st.error("Groq API Key not found. Please add it to Streamlit Secrets.")
    st.stop()


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )


@st.cache_resource
def build_retriever(pdf_path):
    """Heavy, shareable part: load PDF, split, embed, index. Cached."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def build_chain(pdf_path):
    """Per-session chain, so chat memory is not shared between users."""
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        groq_api_key=groq_api_key,
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=build_retriever(pdf_path),
        memory=st.session_state.memory,
    )


# Usage:
# chain = build_chain("your.pdf")
# result = chain.invoke({"question": user_question})
# st.write(result["answer"])
