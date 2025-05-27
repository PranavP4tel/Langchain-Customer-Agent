#Importing Libraries
import streamlit as st
st.set_page_config(page_title = "Eva")

from agent import customer_agent

#setting page configuration and page content
st.title("Eva - Your customer support agent")
st.markdown("#### I can assist you with queries regarding our products or services.")
st.write("FAQs Dataset source: https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset")

#Setting chat history in the session
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("Enter your Gemini API Key \n\n(You can get one from Google's AI Studio)")
    GOOGLE_API_KEY = st.text_input(label = "API Key", type="password")

#Printing the entire message history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

#Taking the user prompt
prompt = st.chat_input("Ask away...")


if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    #Adding user prompt to the chat history
    st.session_state.messages.append({"role":"user","content":prompt})

    #Giving output
    with st.chat_message("assistant"):
        try:
            response = customer_agent(api_key = GOOGLE_API_KEY, agent_input= {"input":prompt})
            st.session_state.messages.append({"role":"assistant", "content": response["output"]})
            st.markdown(response["output"])

        except Exception as e:
            st.markdown(f"Exception: {e}")