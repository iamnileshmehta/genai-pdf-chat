import streamlit as st
import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.chains import ConversationalRetrievalChain


load_dotenv()

if not os.getenv("HF_TOKEN"):
    st.error("Hugging Face Token not found.")
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

    llm = ChatHuggingFace(
        model=HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0,
            token=os.getenv("HF_TOKEN")
        )
    )


    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

    return qa_chain
