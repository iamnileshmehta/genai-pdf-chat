import streamlit as st
import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

from langchain_community.embeddings import HuggingFaceEmbeddings
# Note: LangChain ke standard updates ke mutabik direct import use karein
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# 1. Local ke liye dotenv load karein, agar Streamlit par hai toh Secrets automatic os.environ mein aa jaate hain
load_dotenv()

# Streamlit Cloud par Secrets se direct key uthane ka sasta aur best tarika
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key not found. Please add it to Streamlit Secrets.")
    st.stop()


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )


@st.cache_resource
def build_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    # 2. FIX: model_name aur groq_api_key parameters ko update kiya gaya hai
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",  # Correct standard parameter
        temperature=0,
        groq_api_key=groq_api_key          # Correct standard parameter
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

    return qa_chain
