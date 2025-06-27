# Langchain Customer Agent

Description - An interactive customer agent built with Streamlit, Langchain, Gemini and ChromaDB, aimed at assisting users with their queries by way of a RAG system.
Using a web-based chat interface, the user queries are taken and the agent will search the vector database for the relevant answer, providing them back to the customer.
It also maintains context via a chat history and saves conversation status as Completed (C), Urgent (U) or Incomplete (I), coupling it with the customer ID for follow-ups or further human intervention.

---
Workflow:
1. Interface - A Streamlit based interface that accepts a Gemini API Key to enable the conversation. Saves the chat history in the session and makes function calls to the agent, retrieving and displaying the results.
2. Data Preprocessing & Transformation - Reading the dataset and performing data cleaning. Create a persistent vector store and saving the embedded FAQs for use by the agent.
3. Agent - A Langchain based agent that makes use of:
* Tools created to provide functionality to the agent - retrieval tool to retrieve relevant answers from the vector store and a save tool to save the conversation status.
* Conversation Buffer Memory to save the chat history and provide context of the conversation to the agent.
* Google's Gemini - The core of the agent, coupled with Langchain's prompt template, tool calling agent and agent executor.

<br/>
There is also a 'search' tool created, that can allow the agent to search relevant websites using DuckDuckGo, to answer user queries beyond the scope of the RAG system. As of now, the tool isn't incorporated in the existing workflow, but can be applied to extend the functionality and usability of the agent.

<br/><br/>
There are various areas of improvement in this project and I hope to accomplish them along the way. I would be happy to receive any feedback, suggestions or improvements and inculcate them within the application.

---
References:
1. Dataset: [hf://datasets/MakTek/Customer_support_faqs_dataset/train_expanded.json](https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset)
2. Langchain Documentation
3. Gemini API Key obtained from Google's AI Studio

---
You can view the app here: https://langchain-customer-agent.streamlit.app/
