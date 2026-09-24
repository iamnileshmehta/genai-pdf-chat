import streamlit as st
import os 
from dotenv import load_dotenv
import langchain

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# FIX: ChatHuggingFace wrapper import kiya gaya hai jo conversational task ko natively handle karega
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.callbacks import LangChainTracer

load_dotenv()

hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
langsmith_key = os.getenv("LANGCHAIN_API_KEY") or st.secrets.get("LANGCHAIN_API_KEY")


if not hf_token:
    st.error("Hugging Face Token not found. Please add it to Streamlit Secrets.")
    st.stop()

# --- BULLETPROOF LANGSMITH EXPLICIT INITIALIZATION ---
if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "genai-pdf-chat-groq"
    # LangChain v0.2+
    langchain.turn_on_tracing()

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

    # Base endpoint setup (Novita ke standard constraints bypass karne ke liye text-generation default rakhein)
    raw_llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        huggingfacehub_api_token=hf_token,
        task="text-generation"
    )

    # FIX: Is raw endpoint ko ChatHuggingFace ke andar wrap kiya gaya hai
    # Yeh automatic pipeline ko /v1/chat/completions standard par badal dega aur Novita provider crash nahi karega
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
