import streamlit as st
import asyncio
from langchain import GoogleSerperAPIWrapper
from together import Together
from langchain_together import ChatTogether
from transformers import pipeline
import os
import fitz
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Environment Variables
os.environ["TOGETHER_API_KEY"] = "3c4b2895b1143c3aae675cc061df9d32b47dc697310fc2d4075cc13bcc857252"
os.environ["PINECONE_API_KEY"] = "pcsk_7J5qMz_USujTmoXLnjKr3wJGPwRgUE7DpxdCvM1et2dV3oqz2okgjSAgWj7W4LuQfqZS1n"
os.environ["SERPER_API_KEY"] = "de122d9b8f1338725435eeb734f5fd0961f6182e"

# Ensure asyncio loop compatibility in Streamlit
def create_event_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Initialize Pinecone
create_event_loop()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Use asyncio.run for initializing asynchronous components
async def init_embeddings():
    return PineconeEmbeddings(
        model='multilingual-e5-large',
        pinecone_api_key=os.environ.get('PINECONE_API_KEY')
    )

embeddings = asyncio.run(init_embeddings())

cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Define history variables
history_men = []
query_men = []
message_men = []
history_uni = []

# Text Summarizer
summarizer = pipeline("summarization")
def summarize_history(history):
    if len(history) > 1000:
        concatenated_history = " ".join(history)
        summary = summarizer(concatenated_history, max_length=100, min_length=50, do_sample=False)
        return [summary[0]['summary_text']]
    return history

# Streamlit Page Configuration
st.set_page_config(page_title="Multi-Purpose Chatbot", layout="wide")
st.title("Welcome to the AAFT MENTAL HEALTH CHATBOT WITH MULTI-PURPOSE WORK")
st.sidebar.title("Chatbot Options")
option = st.sidebar.radio("Select a Chatbot:", ["Mental Health Chatbot", "PDF Chatbot", "Universal Search"])

# Helper Functions
def chatbot_men(user_prompt):
    """Handles mental health chatbot queries."""
    if any(word in user_prompt.lower() for word in ["thank you", "thanks"]):
        return "You're welcome! I'm here to help."

    # Search API for mental health queries
    search = GoogleSerperAPIWrapper(gl="in", k=5)
    search_results = search.run(user_prompt)
    
    prompt = f"""Use the context below to generate the response:

    Question: {user_prompt}
    Context: {search_results}
    """

    client = ChatTogether(model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo")
    response = client.invoke([HumanMessage(content=prompt)])
    history_men.append(user_prompt)
    return response.content

def chatbot_uni(user_prompt):
    """Handles universal search chatbot queries."""
    search = GoogleSerperAPIWrapper(gl="in")
    search_results = search.run(user_prompt)
    
    prompt = f"""Use the context below to generate the response:

    Question: {user_prompt}
    Context: {search_results}
    """

    client = Together()
    message = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model="mistralai/Mistral-7B-Instruct-v0.3", messages=message)
    return response.choices[0].message.content

def load_documents_to_pinecone(file_path, namespace, index_name):
    """Loads a PDF's content into Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
    reader = fitz.open(file_path)
    text = "".join([reader[i].get_text() for i in range(len(reader))])
    docs = text_splitter.create_documents([text])
    PineconeVectorStore.from_documents(
        documents=docs,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )
    return text

def chatbot_pdf(user_prompt, index_name, namespace):
    """Handles PDF chatbot queries."""
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    docs = docsearch.similarity_search(user_prompt, k=5)
    llm = ChatTogether(
        openai_api_key=os.environ.get('TOGETHER_API_KEY'),
        model_name='meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
        temperature=0.0
    )
    chain = load_qa_chain(llm)
    response = chain.run(input_documents=docs, question=user_prompt)
    return response

# Chatbot Interfaces
if option == "Mental Health Chatbot":
    st.header("Mental Health Chatbot")
    user_input = st.text_input("Ask a question related to mental health:")
    if user_input:
        response = chatbot_men(user_input)
        st.write("**Response:**", response)

elif option == "PDF Chatbot":
    st.header("PDF Chatbot")
    uploaded_file = st.file_uploader("Upload a PDF:", type="pdf")
    namespace = st.text_input("Enter a unique namespace for this document:")
    if uploaded_file and namespace:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        st.write("Indexing the PDF...")
        load_documents_to_pinecone(uploaded_file.name, namespace, index_name="aryaan123")
        st.success("PDF indexed successfully! You can now ask questions.")
        user_input = st.text_input("Ask a question about the document:")
        if user_input:
            response = chatbot_pdf(user_input, "aryaan123", namespace)
            st.write("**Response:**", response)

elif option == "Universal Search":
    st.header("Universal Search Chatbot")
    user_input = st.text_input("Ask any question:")
    if user_input:
        response = chatbot_uni(user_input)
        st.write("**Response:**", response)
