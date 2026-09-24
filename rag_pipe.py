import streamlit as st
import os 
from dotenv import load_dotenv

# 🔴 Sabse Critical: load_dotenv() se PEHLE variables environment mein hone chahiye
hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
langsmith_key = os.getenv("LANGCHAIN_API_KEY") or st.secrets.get("LANGCHAIN_API_KEY")

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "genai-pdf-chat-groq"

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

if not hf_token:
    st.error("Hugging Face Token not found. Please add it to Streamlit Secrets.")
    st.stop()


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

    raw_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
        task="text-generation"
    )

    llm = ChatHuggingFace(llm=raw_llm)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return qa_chain
