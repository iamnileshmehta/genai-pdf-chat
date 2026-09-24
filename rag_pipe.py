import streamlit as st
import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# FIX: Sahi endpoints import kiye gaye hain jo Pydantic ko crash nahi karenge
from langchain_huggingface import HuggingFaceEndpoint

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Cache aur configs runtime check
load_dotenv()

hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not hf_token:
    st.error("Hugging Face Token not found. Please add it to Streamlit Secrets.")
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

    # FIX: Pydantic v2 compliant standard architecture setup
    # Direct Endpoint use karne par server optimization error nahi aate aur speed badh jaati hai
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        temperature=0.1,
        huggingfacehub_api_token=hf_token, # Sahi parameter syntax mapping
        task="text-generation"             # Model structure allocation
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

    return qa_chain
