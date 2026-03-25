from utils import files_handler, chunk_text, store_chunks, retrieve, build_user_message, groq_client
import streamlit as st

st.title("Welcome to ChatBot!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_files = st.file_uploader("Upload data", accept_multiple_files=True)
if uploaded_files:
    file_contents = files_handler(uploaded_files)
    if file_contents:
        for file_content in file_contents:
            chunks = chunk_text(file_content['content'], 2, 1)
            store_chunks(chunks, file_content)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Enter your prompt"):
    
    retrieved_chunks = []
    try:
        retrieved_chunks = retrieve(prompt)
    except Exception as e:
        st.warning("No documents in database yet")
    
    content = build_user_message(prompt, retrieved_chunks)
    
    st.session_state.messages.append({"role": "user", "content": content})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):        
        llm_response = groq_client(st.session_state.messages)
        st.write(llm_response)
    
    st.session_state.messages.append({"role": "assistant", "content": llm_response})