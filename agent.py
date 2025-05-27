#Importing libraries
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit as st
import pandas as pd


#Creating the intructions for the agent to follow
#This message is to be used in conjunction with the search tool
system_message = """You are a helpful customer service assistant to solve user queries regarding the products and services of ABC Corp.
When answering a user's question, use the tools provided to you.
Make sure to use the 'retrieve' tool first, and if you do not find the answer, then use the 'search' tool
After using a tool, the tool output will be provided back to you. 
Save the conversation status after using the 'save_results' tool after you feel the conversation has ended or you cannot resolve the use query.
Do not answer any irrelevant queries beyond the scope of being a customer service agent.
"""

#This message is used when the search tool is removed
system_message = """You are a helpful customer service assistant to solve user queries regarding the products and services of ABC Corp.
When answering a user's question, use the `retrieve` tool provided to you.
After using a tool, the tool output will be provided back to you. 
Save the conversation status after using the 'save_results' tool after you feel the conversation has ended or you cannot resolve the use query.
Do not answer any irrelevant queries beyond the scope of being a customer service agent.
"""


#Defining the tools to be used by the agent
#@tool 
#def search(search_string: str) -> str:
#  """Use this tool to search the web for additional context.
#  Only use this tool when you do not possess enough information to answer the inputs.
#  Make sure to use search techniques to format your search string.
#  Conduct a maximum of 3 search tries, after which you can quit the search and move along the conversation.'
#  """
  
#  search_object = DuckDuckGoSearchRun(output_format = "json")
#  search_results = search_object.invoke(search_string)
#  return search_results

#Loading the vector store and caching it.
@st.cache_resource
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        collection_name="customer_collection",
        persist_directory="./chroma_embeddings_db",
        embedding_function=embedding_model
    )

vector_store = load_vector_store()

@tool
def retrieve(question: str) -> str:
    """Use this tool to search the vector store for relevant answers to use queries.
    If you that the responses matched the user queries then do not query the vector store again.
    Otherwise, try again with different variations of the question to obtain results.
    Do not go beyond 3 tries of querying the vector store, use search tool."""
  
    retrieved_docs = vector_store.similarity_search(question, k = 3)
    combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return f"Context: \n {combined_context}"

@tool
def save_results(status: str) -> str:
    """
    Use this tool to save the results after the conversation has concluded.
    Make sure to save the conversation status as {"C","I","U"}
    C = Completed conversation, 
    I= Incomplete conversation (when user wants to speak to a human or he is not satisfied with the responses), 
    U = Urgent resolution needed.
    Make sure to understand the conversation you have had with the user, to decide the status to be saved.
    """
    file_path = "Customer_Chat_Results.csv"
    
    df = pd.read_csv(file_path)
    
    # Determine next CustomerID
    if df.empty:
        next_id = 1
    else:
        next_id = int(df["CustomerID"].iloc[-1]) + 1

    # Add new row
    new_row = pd.DataFrame([[next_id, status]], columns=["CustomerID", "Status"])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save to CSV
    df.to_csv(file_path, index=False)
    return "Customer Information Saved"


#tools = [retrieve, search, save_results]
tools = [retrieve, save_results]

#Defining the prompt template with all the components
prompt = ChatPromptTemplate.from_messages([
   ("system", system_message),
   MessagesPlaceholder(variable_name = "chat_history"),
   ("human", "{input}"),
   MessagesPlaceholder(variable_name = "agent_scratchpad")
])

#Defining the memory for the agent
memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)

#Defining the langchain customer agent
def customer_agent(api_key: str, agent_input: dict)-> str:
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash", 
                                 temperature = 0.2, 
                                 google_api_key = api_key
                                 )

    agent = create_tool_calling_agent(
       llm = llm,
       tools = tools,
       prompt = prompt
    )
    
    agent_executor = AgentExecutor(
       agent = agent,
       tools = tools,
       memory = memory,
       verbose = True
    )

    return agent_executor.invoke({"input": agent_input["input"],"chat_history":memory})