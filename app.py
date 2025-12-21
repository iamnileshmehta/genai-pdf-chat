import streamlit as st
import os 
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_classic.memory import ConversationBufferMemory

from langchain_classic.chains import ConversationalRetrievalChain


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#------------------------------------------------------------------------------
#ENV SETUP
#------------------------------------------------------------------------------

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

#------------------------------------------------------------------------------
#STREAMLIT UI
#------------------------------------------------------------------------------

st.set_page_config(page_title="GenAI PDF Chatbot")
st.title("GenAI based Document Reader")
st.write("Upload PDF and asked questions")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


#------------------------------------------------------------------------------
#MEMORY
#------------------------------------------------------------------------------

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

#------------------------------------------------------------------------------
#PIPELINE
#------------------------------------------------------------------------------

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

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory
    )

    return qa_chain

#------------------------------------------------------------------------------
#STREMLIT UI
#------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF loaded")

    qa_chain = build_chain("temp.pdf")

    user_input = st.chat_input("Ask a question about the PDF")

    if user_input and user_input.strip():
        st.chat_message("user").write(user_input)

        with st.spinner("Thinking..."):
            result = qa_chain.invoke({
                "question": user_input
            })

        st.chat_message("assistant").write(result["answer"])

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.memory.clear()
    

