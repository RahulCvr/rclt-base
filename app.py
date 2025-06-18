# SQLite workaround for Streamlit Cloud
import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

#path to the chroma database
CHROMA_PATH = "chroma"

#prompt template for the chatbot
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Page config
st.set_page_config(
    page_title="rclt.ai",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_embedding_function():
    """Initialize and cache the embedding function."""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'],  # Replace with your deployment name
        openai_api_version="2024-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_EMBEDDING_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_EMBEDDING_API_KEY']
    )

@st.cache_resource
def get_vector_db():
    """Initialize and cache the vector database."""
    embedding_function = get_embedding_function()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

@st.cache_resource
def get_chat_model():
    """Initialize and cache the chat model."""
    return AzureChatOpenAI(
        azure_deployment=os.environ['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME'],  # Replace with your deployment name
        openai_api_version="2024-12-01-preview",
        azure_endpoint=os.environ['AZURE_OPENAI_CHAT_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_CHAT_API_KEY'],
    )

# Initialize components with error handling
try:
    embedding_function = get_embedding_function()
    db = get_vector_db()
    chat_model = get_chat_model()
except KeyError as e:
    st.error(f"Missing environment variable: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error initializing components: {e}")
    st.stop()

def query_knowledge_base(query_text):
    """Query the knowledge base and return response with sources."""
    try:
        # Search the DB
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        if len(results) == 0:
            return "I couldn't find any relevant information in the knowledge base.", []
        
        # Prepare context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get response from chat model
        response = chat_model.invoke(prompt)
        response_text = response.content
        
        # Extract sources
        sources = [doc.metadata.get("source", "Unknown") for doc, _score in results]
        
        return response_text, sources
        
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}", []

# Main UI
st.title("rclt.ai")
st.markdown("Your Second Brain")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("This chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on your knowledge base.")
    st.write("Upload your files to the /data folder and run the create_database.py script to create the database.")
    
    st.header("How it works")
    st.write("1. Your question is converted to embeddings")
    st.write("2. Similar documents are retrieved from ChromaDB")
    st.write("3. The LLM Model generates an answer based on the context")

    st.header("Default Knowledge Base")
    st.write("The default knowledge base contains the book 'Alice in Wonderland' by Lewis Carroll.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"{i}. {source}")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, sources = query_knowledge_base(prompt)
        
        st.markdown(response_text)
        
        if sources:
            with st.expander("Sources"):
                for i, source in enumerate(sources, 1):
                    st.write(f"{i}. {source}")
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources
    })

# Footer
st.markdown("---")
st.markdown("*Made by Rahul Reddy CV*") 