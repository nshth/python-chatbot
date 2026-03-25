import os
import sys
import filetype
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from groq import Groq
import base64
import re
from sentence_transformers import SentenceTransformer
import chromadb
import streamlit as st
from io import StringIO
import io
from dotenv import load_dotenv

load_dotenv("./.env")
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
client = chromadb.PersistentClient(path="./chroma_db")
# client = chromadb.Client()
model = SentenceTransformer("all-MiniLM-L6-v2")

message_history: list[dict[str, str]] = []

def preprocessor_txt(file_path: str) -> str:
    with open(file=file_path, mode="r") as f:
        content = f.read()
    return content


def preprocessor_pdf(file_path) -> str:
    reader = PdfReader(file_path)
    content = ""

    for page in reader.pages:
        content += page.extract_text() or "" + "\n"

    if not content.strip():
        pages = convert_from_path(file_path, poppler_path=r"C:/poppler/Library/bin")
        for page in pages:
            content += pytesseract.image_to_string(page) + "\n"

    return content

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> str:
    chunks = []

    sentences = re.split(r'(?<=[.!?])\s+', text)

    step = chunk_size - chunk_overlap

    for i in range(0, len(sentences), step):
        window = sentences[i:i + chunk_size]  
        chunk = " ".join(window)
        chunks.append(chunk)

    return chunks

def embed(chuncks: str):
    vector = model.encode(chuncks).tolist()
    return vector

def is_already_stored(file_path: str) -> bool:
    collection = client.get_or_create_collection("documents")
    results = collection.get(ids=[file_path + "0"])  
    return bool(results["ids"])

def store_chunks(chunks: list[str], processed_content) -> None:
    if is_already_stored(processed_content['path']):
        print("skipping...")
        return
    collection = client.get_or_create_collection("documents")
    for i, chunk in enumerate(chunks):
        embedding = embed(chunk)
        collection.add(
            documents=[chunk],
            embeddings=embedding,
            ids=processed_content['path'] + str(i) 
        )
    return collection

def retrieve(query: str, n_results: int = 2) -> list[str]:
    collection = client.get_or_create_collection("documents")
    query_embedding = embed(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]

def preprocessor_image(image_path) -> str | None:
  with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded

def files_handler(uploaded_files: list) -> list[dict]:
    processed_files = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getbuffer()
        file_name = uploaded_file.name
        
        # Detect file type from bytes
        kind = filetype.guess(io.BytesIO(file_bytes))

        if kind is None:
            try:
                content = bytes(file_bytes).decode('utf-8')
                processed_files.append({
                    "path": file_name,  # Just the name, not a real path
                    "type": "text",
                    "content": content
                })
            except UnicodeDecodeError:
                st.warning(f"Could not decode {file_name}")
            continue

        mime = kind.mime

        if mime == "application/pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            content = ""
            for page in reader.pages:
                content += page.extract_text() or "" + "\n"
            processed_files.append({
                "path": file_name,
                "type": "text",
                "content": content
            })

        elif mime in ["image/jpeg", "image/png"]:
            encoded = base64.b64encode(file_bytes).decode('utf-8')
            ext = kind.extension
            processed_files.append({
                "path": file_name,
                "type": "image",
                "content": encoded,
                "format": ext
            })

        else:
            st.warning(f"Unsupported file type: {mime}")

    return processed_files

def exit_app(user_message: str) -> None:
    if user_message == "q":
        sys.exit()

def build_user_message(user_prompt: str, retrieved_chunks: list[dict]) -> list[dict]:
    content = []

    content.append({
        "type": "text",
        "text": f"Uploaded document content:\n{retrieved_chunks}"
    })
    # Add user's query
    content.append({
        "type": "text",
        "text": f"User query: {user_prompt}"
    })

    return content

def groq_client(message_history):
    client = Groq(
        api_key=os.environ.get("GROQ_API"))

    chat_completion = client.chat.completions.create(
        messages=message_history,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    decoded_response = chat_completion.choices[0].message.content
    return decoded_response


# handle image differently
# metadat filteirng?
# score threshold?
# Reranking? model to rerank them by relevance or retreived output
# Hybrid search? vector search with keyword matching
# Evaluation?