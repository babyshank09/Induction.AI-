import streamlit as st 
import os 
from langchain_groq import ChatGroq 
from langchain_openai import OpenAIEmbeddings 
from langchain_community.embeddings import OllamaEmbeddings  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain 
from langchain_core.prompts import ChatPromptTemplate  
from langchain_community.document_loaders import PyPDFDirectoryLoader  
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import UnstructuredPowerPointLoader
import dotenv
from dotenv import load_dotenv 

st.title("Induction.AI") 
st.subheader("A Chatbot for any onboarding related query")  

st.sidebar.title("Settings") 
groq_api_key= st.sidebar.text_input("Enter your Groq API key", type="password")
openai_api_key= st.sidebar.text_input("Enter your OpenAI API key:", type="password")

if groq_api_key:
    llm= ChatGroq(model="Gemma2-9b-It", groq_api_key= groq_api_key) 

prompt= ChatPromptTemplate.from_template(
    """ 
    "You are a helpful assistant. Answer the questions based on the given context only 
    <context> 
    "{context}" 
    </context> 
    Question: "{input}" 
    """
)


uploaded_files = st.sidebar.file_uploader(
    "Choose PowerPoint files", 
    type=["ppt", "pptx"], 
    accept_multiple_files=True
)

if uploaded_files:
    all_documents = []  
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())  #

        loader = UnstructuredPowerPointLoader(uploaded_file.name)
        documents = loader.load()
        all_documents.extend(documents)

def create_vector_embedding(final_docs, openai_api_key): 
    if "vectorstore" not in st.session_state: 
        st.session_state.text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
        st.session_state.final_docs= st.session_state.text_splitter.split_documents(final_docs) 
        st.session_state.embedder= OpenAIEmbeddings(openai_api_key= openai_api_key) 
        st.session_state.vectorstore= FAISS.from_documents(st.session_state.final_docs, embedding= st.session_state.embedder) 
 

if st.button("Embed Document"): 
    create_vector_embedding(all_documents, openai_api_key) 
    st.write("Vector Database is ready")  

query= st.text_input("Enter the query from the onboarding documentation") 

if query: 
    document_chain= create_stuff_documents_chain(llm,prompt)   
    retriever= st.session_state.vectorstore.as_retriever() 
    retrieval_chain= create_retrieval_chain(retriever, document_chain)  

    response= retrieval_chain.invoke({"input": query})
    st.write(response["answer"])  

    with st.expander("Document Similarity Search"): 
        for i, doc in enumerate(response["context"]): 
            st.write(doc.page_content)
            st.write("------------------------------") 
