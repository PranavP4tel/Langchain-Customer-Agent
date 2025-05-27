#Importing libraries
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np

#Reading the dataset and some preprocessing
df = pd.read_json("hf://datasets/MakTek/Customer_support_faqs_dataset/train_expanded.json", lines=True)
df.drop_duplicates(inplace = True)
df["Q&A"] = df["question"] + " Answer: "+df["answer"]

#Creating the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#Creating the vector store and persisting it so that it can be used directly
vector_store = Chroma(
    collection_name ="customer_collection",
    embedding_function = embedding_model,
    persist_directory = "./chroma_embeddings_db"
)

#Adding all the FAQs in the vector store
vector_store.add_texts(ids = [str(i) for i in range(len(df))], texts = list(df["Q&A"]))
print("Vector store created!")