# importing libraries
import streamlit as st
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import json


st.title("Startup Guide")
# google trends api
from pytrends.request import TrendReq
@st.cache_data
def fetch_google_trends_data(keyword="startups"):
    pytrends = TrendReq()
    pytrends.build_payload([keyword], cat=0, timeframe='now 7-d', geo='US')
    trends = pytrends.interest_over_time()
    return trends.reset_index().to_dict(orient='records')

# news api key
import requests
@st.cache_data
def fetch_newsapi_data(query="startups"):
    api_key = st.secrets["NEWS_API_KEY"]
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    return []

# combine all data 
from langchain.schema import Document

# Combine all data and convert to LangChain Document objects
def load_combined_data():
    google_data = fetch_google_trends_data("startups")
    news_data = fetch_newsapi_data("startups")

    combined_docs = []

    # Convert Google Trends data to Document objects
    for item in google_data:
        content = f"Date: {item['date']}, Interest: {item['startups']}"
        combined_docs.append(Document(page_content=content))

    # Convert NewsAPI data to Document objects
    for article in news_data:
        content = f"Title: {article['title']}, Description: {article['description']}"
        combined_docs.append(Document(page_content=content))

    return combined_docs


# Split the loaded data into chunks
@st.cache_data
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    return text_splitter.split_documents(_data)

# Create a vector store using FAISS
@st.cache_resource
def create_vector_store(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process data
data = load_combined_data()
docs = split_data(data)
vectorstore = create_vector_store(docs)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

# Defining the prompt template for DeepSeek R1
system_prompt = (
    "You are an AI Startup Consultant specializing in market research, idea validation, and business insights. "
    "Use the provided context to answer user queries related to startup ideas, trends, competitors, and growth potential. "
    "If unsure, respond with 'I don't know' instead of guessing. "
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# User input and response generation
query = st.text_input("🗣️ Enter your query:")
if st.button("submit"):
       question_answer_chain = create_stuff_documents_chain(llm, prompt)
       rag_chain = create_retrieval_chain(retriever, question_answer_chain)
       response = rag_chain.invoke({"input": query})
       st.write(response["answer"])

