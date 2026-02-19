import os
import sys
from pathlib import Path
from pypdf import PdfReader
from groq import Groq
import base64
from dotenv import load_dotenv

load_dotenv("./.env")
openrouter_secret = os.getenv("API")
# grok_secret = os.getenv("GROK_API")

message_history: list[dict[str, str]] = []

def preprocessor_txt(file_path: str) -> str:
    with open(file=file_path, mode="r") as f:
        content = f.read()
    return content

def preprocessor_pdf(file_path) -> str:
    reader = PdfReader(file_path)
    content = ""

    for page in reader.pages:
        content += page.extract_text() + "\n"

    return content

def preprocessor_image(image_path):
  with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded

def files_handler(file_paths: list[str]) -> list[dict]:
    processed_files = []

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"\nWARN: File {file_path} does not exist!\n")
            continue

        file_type = Path(file_path).suffix.lower()

        if file_type == ".txt":
            content = preprocessor_txt(file_path)
            processed_files.append({
                "path": file_path,
                "type": "text",
                "content": content
            })

        elif file_type == ".pdf":
            content = preprocessor_pdf(file_path)
            processed_files.append({
                "path": file_path,
                "type": "text",
                "content": content
            })

        elif file_type in [".jpg", ".jpeg", ".png"]:
            content = preprocessor_image(file_path)
            processed_files.append({
                "path": file_path,
                "type": "image",
                "content": content,
                "format": file_type.replace(".", "")
            })

        else:
            print(f"\nWARN: Unsupported file type {file_type}\n")

    return processed_files


# def file_content_prompt_generator(file_contents: dict[str, str]) -> str:
#     prompt_starter = "The user has uploaded some files. Here are the contents..."

#     for key, val in file_contents.items():
#         prompt_starter += f"\n\nFile: {key}\n```\n{val}\n```\n Query:"
#     return prompt_starter


def exit_app(user_message: str) -> None:
    if user_message == "q":
        sys.exit()

def build_user_message(user_prompt: str, files: list[dict]) -> list[dict]:
    content = []

    # Add file contents
    for file in files:
        if file["type"] == "text":
            content.append({
                "type": "text",
                "text": f"Uploaded document content:\n{file['path']} \n\n{file['content']}"
            })

        elif file["type"] == "image":
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{file['format']};base64,{file['content']}"
                }
            })

    # Add user's query
    content.append({
        "type": "text",
        "text": f"User query: {user_prompt}"
    })

    return content

# def openrouter_client(message_history) -> str:
#     api_url = "https://openrouter.ai/api/v1/chat/completions"
#     llm_model = "openai/gpt-5.2"
#     message_data = {"model": llm_model, "messages": message_history}
    
#     response = httpx.post(
#         url=api_url,
#         headers={"Authorization": f"Bearer {openrouter_secret}"},
#         json=message_data  
#     )

#     # Check status code BEFORE parsing JSON
#     if response.status_code >= 400:
#         print(f"\nApp ran into an error. Status code {response.status_code}")
#         sys.exit(1)

#     response_data = response.json()
#     decoded_response = response_data["choices"][0]["message"]["content"]

#     return decoded_response


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
        content = build_user_message(prompt, file_contents)

        # if len(content) > 0:
        #     file_prompt_prefix = file_content_prompt_generator(content)
        #     prompt = file_prompt_prefix + prompt + '\n```\n'
        
        message_history.append({"role": "user", "content": content})

        llm_response = groq_client(message_history)

        message_history.append({"role": "assistant", "content": llm_response})

        print("AI:", llm_response)


if __name__ == "__main__":
    main()
