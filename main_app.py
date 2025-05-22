# importing libraries
import streamlit as st
import requests
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import json

# page configuration
st.title("📊 MarketMinds: Startup Guide")
with st.sidebar:
    st.image("Startup-Guide.png", use_column_width=True)
    st.markdown(
               "Validate your business ideas with AI-powered market insights in real-time."
    )

    st.title("Configuration")
    temp = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.3)
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

# if using deepseek through ollama
# llm = ChatOllama(
#     model="deepseek-r1:1.5b",
#     temperature=0.3,
#     base_url="http://localhost:11434"
# )

# LLM and retriever
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=temp,
    max_tokens=None,
    timeout=None
)

# Defining the prompt template for DeepSeek R1
system_prompt = (
    "You are an AI Startup Consultant specializing in market research, idea validation, and business insights. "
    "Use the provided context to answer user queries related to startup ideas, trends, competitors, and growth potential. "
    "If user ask any question related to his/ her starup please"
    "answer his/ her queries according to the data provided to you."
    "If unsure, respond with 'I don't know' instead of guessing.  "
    "\n\n"
    "If the user sends a greeting (like 'hi', 'hello', 'hey' , or 'what you can do'), respond with a friendly greeting, "
    "introduce yourself as a startup guider, and let them know you're available to assist with any questions about your startup idea. "
    "Also, ask: 'How can I help you today?'"
    "please only greet them once and then only give answer to queries and don't introduce yourself with every answer\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# session state for chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# User input and response generation
query = st.chat_input("🗣️ Enter your business idea")
if query:
       st.chat_message("user").write(query)
       question_answer_chain = create_stuff_documents_chain(llm, prompt)
       rag_chain = create_retrieval_chain(retriever, question_answer_chain)
       response = rag_chain.invoke({"input": query})
       answer=response["answer"]

        # display and store
       st.chat_message("assistant").write(answer)
       st.session_state.messages.append({"role": "user", "content": query})
       st.session_state.messages.append({"role": "assistant", "content": answer})

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask Question"}]
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

