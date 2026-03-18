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

def files_handler(file_paths: list[str]) -> list[dict]:
    processed_files = []
    # mime_type_list = ["image/jpeg", "image/png"]
    # robot.txt

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"\nWARN: File {file_path} does not exist!\n")
            continue

        kind = filetype.guess(file_path)

        if kind is None: 
            try:
                content = preprocessor_txt(file_path)
                processed_files.append({
                    "path": file_path,
                    "type": "text",
                    "content": content
                })
            except UnicodeDecodeError:
                print(f"\nWARN: Could not determine file type for {file_path}\n")
            continue


        mime = kind.mime

        if mime == "application/pdf":
            content = preprocessor_pdf(file_path)
            processed_files.append({
                "path": file_path,
                "type": "text",
                "content": content
            })

        elif mime in ["image/jpeg", "image/png"]:
            content = preprocessor_image(file_path)
            ext = kind.extension
            processed_files.append({
                "path": file_path,
                "type": "image",
                "content": content,
                "format": ext
            })

        else:
            print(f"\nWARN: Unsupported file type {type}\n")

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

def main() -> None:
    print("Welcome to ChatBot!\n\n\n")
    while True:
        prompt = input("Prompt: ")
        exit_app(prompt)

        file_upload_paths: list[str] = []  # Create fresh list each iteration

        file = input("Do you want to upload files? (y or n) ")
        while True:
            if file == 'y':
                _file_path = input("File location: ")
                if _file_path == "":
                    break
                else:
                    file_upload_paths.append(_file_path)
            else:
                break

        file_contents = files_handler(file_upload_paths)
        retrieved_chunks = []
        if file_contents:
            for file_content in file_contents:
                chunks = chunk_text(file_content['content'], 2, 1)
                store_chunks(chunks, file_content)
            
        try:
            retrieved_chunks = retrieve(prompt)
        except Exception as e:
            retrieved_chunks = []
            print("No documents in database yet") 

        print("RETRIEVED CHUNKS:", retrieved_chunks)
        content = build_user_message(prompt, retrieved_chunks)

        message_history.append({"role": "user", "content": content})

        llm_response = groq_client(message_history)

        message_history.append({"role": "assistant", "content": llm_response})

        print("AI:", llm_response)


if __name__ == "__main__":
    main()


# handle image differently
# metadat filteirng?
# score threshold?
# Reranking? model to rerank them by relevance
# Hybrid search? vector search with keyword matching
# Evaluation?