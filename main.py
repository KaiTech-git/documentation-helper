from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("Langchain Doc Helper Bot")
prompt = st.text_input("Prompt", placeholder= "Enter your prompt here...")

if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []

def create_source_string(sources:set[str])->str:
    if not sources:
        return ""
    sources_list = list(sources)
    sources_format = "Source: \n"
    for i, url in enumerate(sources_list):
        sources_format +=  f"{i+1}. {url} \n"
    return sources_format

if prompt:
    with st.spinner("Generating response..."):
        generate_response = run_llm(query= prompt)
        sources = set([doc.metadata["source"] for doc in generate_response["context"]])
        formatted_response = (f"{generate_response["answer"]} \n\n {create_source_string(sources)}")
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        
    if st.session_state["chat_answers_history"]:
        for user_query, generated_response in zip(st.session_state["user_prompt_history"],st.session_state["chat_answers_history"]):
            message(user_query,is_user= True)
            message(generated_response,is_user=False)
    
