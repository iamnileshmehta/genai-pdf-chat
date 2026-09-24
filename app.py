import streamlit as st
import os
from rag_pipe import build_chain

st.set_page_config(page_title="GenAI PDF Chatbot", layout="centered")
st.title("📄 GenAI PDF Chatbot")
st.write("Upload a PDF document and ask questions instantly!")

# Initializing global messages tracking block
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploading trigger layer
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Creating a secure temporary storage path
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_file_path = os.path.exists(temp_dir)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner("Processing PDF data through FAISS and Vector Embedding..."):
        try:
            # Building standard architecture chain
            qa_chain = build_chain(temp_file_path)
            st.success("PDF processing complete! Pipeline ready.")
            
            # Simple Interface chat render loops
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("Ask a question about the PDF contents:"):
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.spinner("Retrieving from FAISS and waiting for response..."):
                    response = qa_chain.invoke({"question": prompt})
                    answer = response.get("answer", "Unable to extract response validation constraints.")
                    
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"Execution Error context missing.")
            st.stop()
