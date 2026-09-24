import streamlit as st
import os
from rag_pipe import build_chain
from langchain_core.callbacks import LangChainTracer

st.set_page_config(page_title="GenAI PDF Chatbot", layout="centered")
st.title("📄 GenAI PDF Chatbot")
st.write("Upload a PDF document and ask questions instantly!")

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Processing PDF data through FAISS and Vector Embedding..."):
        try:
            qa_chain = build_chain(temp_file_path)
            st.success("PDF processing complete! Pipeline ready.")
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("Ask a question about the PDF contents:"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.spinner("Retrieving from FAISS and waiting for response..."):
                    
                    # FIX: Explicit manual callback injection for LangSmith
                    callbacks = []
                    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
                    if langsmith_key:
                        tracer = LangChainTracer(project_name="genai-pdf-chat-groq")
                        callbacks = [tracer]
                    
                    # System execution with explicit telemetry callbacks
                    response = qa_chain.invoke({"question": prompt}, {"callbacks": callbacks})
                    answer = response.get("answer", "Unable to extract response validation constraints.")
                    
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"Actual Backend Error: {str(e)}")
            st.stop()
