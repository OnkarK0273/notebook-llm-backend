from fastapi import FastAPI,File, UploadFile, Form
from openai import OpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True, 
    allow_methods = ["*"],
    allow_headers=["*"]
)

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@app.post("/upload_file")
async def upload_file(file:UploadFile = File(...),message:str = Form(...)):
    upload_folder_location = Path(__file__).parent / "upload" 
    upload_folder_location.mkdir(exist_ok=True)
    
    file_location = upload_folder_location / f"{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(await file.read())

    loader = TextLoader(file_location)
    doc = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 450, 
        chunk_overlap = 100, 
        separators = ["\n\n", "\n",' '," ","."]
    )

    chunks =text_splitter.split_documents(documents=doc) 

    embedder = GoogleGenerativeAIEmbeddings(
        model = "models/embedding-001",
        google_api_key = os.getenv("GEMINI_EMBED_SECRET_KEY"),
    )

    vector_store = QdrantVectorStore.from_documents(
        documents = [],
        url="http://localhost:6333",
        collection_name = f"{file_location.stem}_embds_collections",
        embedding = embedder
    )

    vector_store.add_documents(documents = chunks)

    retriever = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name = f"{file_location.stem}_embds_collections",
        embedding = embedder
    )

    relevant_chunks = retriever.similarity_search(
        query = message
    )

    FILE_PROMPT = f"""
        You are an AI assistant who can skilled and experience in analysing and understanding the given chunks of text.
        You're job is to read, analyse and understands the given chunks based on the query given by the user. 

        The chunks that are given to you are already embedded from the vectors. All you have to do is to carfully analyse and understands these chunks,
        and then give a response to the user's query based on your analysis and understanding.

        The chunks are given below:
        {relevant_chunks}

        The Rules you have to follow when giving responses:
        - You should first read, analyse and understand the chunks given to you.
        - Then you will read, analyse and understand the query that user gave. You will understand that what user is actually asking.
        - Then you will give a response to the user based on you analysis and understanding of the chunks you are given and the query that user gave.

    """

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role":"system" , "content": FILE_PROMPT},
            {"role":"user", "content": message}
        ]
    )
    
    return {"user":{"message":message, "file":f"{file.filename}"},"ai": response.choices[0].message.content}