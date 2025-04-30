from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

file_path = Path(__file__).parent / "segmentor.docx"

loader =TextLoader(file_path)
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
)
    
chunks = text_splitter.split_documents(documents=doc)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GEMINI_SECRET_KEY")
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name=f"{file_path.stem}_embeddings_collections",
    embedding=embedder
)

vector_store.add_documents(documents=chunks)

print("Documents have been added to the vector store.")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name=f"{file_path.stem}_embeddings_collections",
    embedding=embedder
)

search_results = retriever.similarity_search(
    query="What is the purpose of the document?",
)

print("Relevant documents", search_results)