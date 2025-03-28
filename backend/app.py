"""
RAG Chatbot for Aviation Data Querying

This project builds a chatbot that allows users to query aviation data
(e.g., bird strikes, minor accidents, incidents) and generates reports
using retrieval-augmented generation (RAG). It uses:
  - ChromaDB for vector storage,
  - LangChain for AI-driven responses,
  - FastAPI for backend endpoints,
  - Next.js (or any frontend) for user interaction.

The code is structured into several components:
  1. Configuration & API Keys
  2. Custom Embeddings (document conversion to embeddings)
  3. Custom Chat Model (LLM-driven responses)
  4. Document Loading and Vector Store management
  5. Querying the vector store with RAG
  6. FastAPI endpoints for chat and re-indexing
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import requests
from datetime import datetime, timedelta
import uvicorn
import json
from typing import List, Dict, Any, Optional

# FastAPI and Pydantic imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, PrivateAttr

# LangChain components (ensure you have the correct versions installed)
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.prompts import PromptTemplate

# Document loaders and text splitters
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

################################################################################
# 1. CONFIGURATION & API KEYS
################################################################################

# Replace these with your own API keys or load from environment variables.
# For example, to use OpenAI, set your OPENAI_API_KEY in your environment.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # <-- Insert your API key here if not using env variables.
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY is not set. Please set the environment variable or update the code.")

# Model names for embedding and chat completions (change if needed)
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"  # Changed from gpt-4 to gpt-3.5-turbo

################################################################################
# 2. CUSTOM EMBEDDINGS CLASS (for document ingestion)
################################################################################

class NormalEmbeddings(Embeddings):
    """
    This class converts texts into embeddings using a normal API call.
    Here we use the OpenAI embedding endpoint.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            try:
                embedding = self.embed_query(text)
                print(f"Successfully embedded text {i+1}/{len(texts)} - Embedding size: {len(embedding)}")
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding text {i+1}: {str(e)}")
                raise

        if not embeddings:
            raise ValueError("No embeddings were generated")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        payload = {
            "model": OPENAI_EMBEDDING_MODEL,
            "input": text
        }
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )
        if response.status_code == 200:
            response_data = response.json()
            # OpenAI returns a list of embeddings in the 'data' field.
            if "data" not in response_data or not response_data["data"]:
                raise ValueError("No embedding data returned")
            embedding = response_data["data"][0].get("embedding")
            if not embedding or not isinstance(embedding, list):
                raise ValueError("Invalid embedding format received")
            return embedding
        else:
            raise Exception(f"Embedding request failed with status {response.status_code}: {response.text}")

################################################################################
# 3. CUSTOM CHAT MODEL CLASS (for generating responses)
################################################################################

class NormalChatModel(BaseChatModel):
    """
    This class wraps a normal API call for chat completions using OpenAI's API.
    """
    model_name: str = Field(default=OPENAI_CHAT_MODEL)
    temperature: float = Field(default=0.0)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
        # Format messages as required by the OpenAI API.
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        if stop:
            payload["stop"] = stop
            
        print(f"Sending request to OpenAI with messages: {formatted_messages}")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        print(f"OpenAI API Response Status: {response.status_code}")
        print(f"OpenAI API Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            # Create a ChatGenerationChunk with the correct format
            message = AIMessageChunk(content=content)
            generation = ChatGenerationChunk(
                message=message,
                generation_info={"finish_reason": "stop"},
                text=content
            )
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"Chat completion request failed with status {response.status_code}: {response.text}")

    @property
    def _llm_type(self) -> str:
        return "normal-chat"

################################################################################
# 4. VECTOR STORE & DOCUMENT LOADING / INDEXING FUNCTIONS
################################################################################

# Initialize our custom models
embeddings = NormalEmbeddings()
llm = NormalChatModel()

# Initialize the vector store (Chroma)
vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

def load_documents_from_directory(directory_path: str):
    """
    Load documents from the specified directory using various file loaders.
    Supports .txt, .md, .pdf, and .docx files.
    """
    loaders = {
        '.txt': TextLoader,
        '.md': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader
    }
    
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in loaders:
                try:
                    loader = loaders[file_extension](file_path)
                    documents.extend(loader.load())
                    print(f"Successfully loaded {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    return documents

def add_to_vector_store():
    """
    Load documents from both a list of URLs and a local data folder,
    split them into chunks, add metadata, and store their embeddings in Chroma.
    Only adds new documents if they don't already exist in the store.
    """
    # Check if vector store is empty
    if vector_store._collection.count() > 0:
        print("Vector store already contains documents. Skipping document addition.")
        return {"status": "success", "message": "Vector store already initialized"}

    # URLs to load (these can be changed to relevant aviation data URLs)
    url_list = [
        "https://python.langchain.com/docs/introduction/",
        "https://www.freecodecamp.org/news/get-started-with-hugging-face/",
        "https://github.com/MODSetter/SurfSense/blob/main/README.md"
    ]
    
    # Load documents from URLs
    loader = WebBaseLoader(
        url_list,
        requests_per_second=2
    )
    url_docs = loader.load()

    # Load local documents from the "./data" folder
    local_docs = load_documents_from_directory("./data")
    all_docs = url_docs + local_docs

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=124,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(all_docs)

    # Add metadata to each chunk
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        source_type = "url" if source.startswith(("http://", "https://")) else "file"
        chunk.page_content = f'''
        <document_chunk>
            <document_source>{source}</document_source>
            <document_type>{source_type}</document_type>
            <document_chunk_content>{chunk.page_content}</document_chunk_content>
        </document_chunk>
        '''
    
    print(f"Adding {len(chunks)} chunks to vector store...")
    vector_store.add_documents(documents=chunks)
    return {"status": "success", "message": f"Added {len(chunks)} chunks to vector store"}

################################################################################
# 5. QUERYING THE VECTOR STORE (RETRIEVAL-AUGMENTED GENERATION)
################################################################################

def query_vector_store(query: str) -> Dict[str, Any]:
    """
    Given a user query, retrieve the top relevant document chunks from the vector store,
    combine them as context, and call the LLM (via OpenAI's API) to generate a response.
    """
    try:
        # Retrieve similar documents/chunks (using a k-nearest search)
        docs = vector_store.similarity_search(query, k=3)  # Reduced from 20 to 3 for better context
        
        concatenated_content = ""
        sources = []
        for doc in docs:
            concatenated_content += doc.page_content + "\n"
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)

        # Construct the system and user messages for the LLM prompt
        system_message = (
            "You are an aviation safety expert assistant. Your task is to answer questions based on the provided context. "
            "Always cite your sources when providing information. If the context doesn't contain enough information to answer "
            "the question, say so clearly. Keep your responses concise and focused on aviation safety."
        )

        user_message = f"""
        Context:
        {concatenated_content}
        
        Question: {query}
        """

        # Use the chat model to generate a response
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        try:
            response = llm.invoke(messages)
            if hasattr(response, 'generations') and response.generations:
                # Extract just the content from the response
                content = response.generations[0].text
                # Clean up any metadata or formatting
                content = content.replace("content='", "").replace("'", "")
                content = content.replace('content="', '').replace('"', '')
                content = content.split(" additional_kwargs")[0].strip()
                content = content.split(" response_metadata")[0].strip()
                content = content.split(" id=")[0].strip()
                
                return {
                    "response": content,
                    "sources": sources
                }
            else:
                content = str(response)
                # Clean up any metadata or formatting
                content = content.replace("content='", "").replace("'", "")
                content = content.replace('content="', '').replace('"', '')
                content = content.split(" additional_kwargs")[0].strip()
                content = content.split(" response_metadata")[0].strip()
                content = content.split(" id=")[0].strip()
                
                return {
                    "response": content,
                    "sources": sources
                }
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while generating the response. Please try again.",
                "sources": sources
            }
    except Exception as e:
        print(f"Exception in query_vector_store: {str(e)}")
        raise

################################################################################
# 6. FASTAPI ENDPOINTS FOR BACKEND-TO-FRONTEND INTERACTION
################################################################################

# FastAPI request/response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []

# Initialize the FastAPI app
app = FastAPI(title="Aviation Data Chatbot")

# Configure CORS (allow all origins for now; adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    On startup, check for new documents and add them to the vector store.
    """
    try:
        # Check if the vector store directory exists
        if not os.path.exists("./chroma_langchain_db"):
            print("Vector store not found. Initializing...")
            add_to_vector_store()
            print("Vector store initialized successfully")
        else:
            print("Vector store exists. Checking for new documents...")
            # Get list of existing documents in the vector store
            existing_docs = vector_store.get()['ids']
            print(f"Found {len(existing_docs)} existing documents in vector store")
            
            # Load and process new documents
            new_docs = load_documents_from_directory("./data")
            if new_docs:
                print(f"Found {len(new_docs)} documents in data directory")
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1024,
                    chunk_overlap=124,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = text_splitter.split_documents(new_docs)
                
                # Add metadata to chunks
                for chunk in chunks:
                    source = chunk.metadata.get("source", "Unknown")
                    source_type = "url" if source.startswith(("http://", "https://")) else "file"
                    chunk.page_content = f'''
                    <document_chunk>
                        <document_source>{source}</document_source>
                        <document_type>{source_type}</document_type>
                        <document_chunk_content>{chunk.page_content}</document_chunk_content>
                    </document_chunk>
                    '''
                
                # Add new chunks to vector store
                if chunks:
                    print(f"Adding {len(chunks)} new chunks to vector store...")
                    vector_store.add_documents(documents=chunks)
                    print("New documents added successfully")
                else:
                    print("No new documents to add")
            else:
                print("No documents found in data directory")
    except Exception as e:
        print(f"Error during startup: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint: Accepts a user message, queries the vector store with RAG,
    and returns the generated response along with source information.
    """
    try:
        print(f"Received chat request with message: {request.message}")
        result = query_vector_store(request.message)
        return ChatResponse(
            response=result["response"],
            sources=result["sources"]
        )
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def reindex_documents():
    """
    Endpoint to manually trigger re-indexing of documents.
    """
    try:
        result = add_to_vector_store()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

################################################################################
# 7. MAIN ENTRY POINT
################################################################################

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
