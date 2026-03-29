import chainlit as cl
from utils import groq_client
import asyncio
from utils import files_handler, chunk_text, store_chunks, retrieve, build_user_message, groq_client


@cl.on_chat_start
async def start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )
    
    files = await cl.AskFileMessage(
        content="Please upload a text file to begin!",
        accept=["text/plain", "text/csv", "application/pdf"],
        max_size_mb=50,
        timeout=180,
        max_files=3
    ).send()
    
    if files is not None and len(files) > 0:
        msg = cl.Message(content="Processing file contents...")
        await msg.send()
        
        file_contents = files_handler(files)
        if file_contents:
            for file_content in file_contents:
                chunks = chunk_text(file_content['content'], 2, 1)
                store_chunks(chunks, file_content)
        
        await msg.update()
        await cl.Message(content="Processed files.").send()
    else:
        await cl.Message(content="No files uploaded. Please upload a file.").send()


@cl.on_message
async def main(message: cl.Message):
    files = [element for element in message.elements if isinstance(element, cl.File)] #???
    if files:
        processed_files = files_handler(files)
        prompt = message.content + processed_files
    else:    
        prompt = message.content
        
    retrieved_chunks = []
    
    try:
        retrieved_chunks = retrieve(prompt)
    except Exception as e:
        await cl.Message(content="No documents in database yet").send()
        return
    
    message_history = cl.user_session.get("message_history")

    content = build_user_message(prompt, retrieved_chunks)

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": content})

    msg = cl.Message(content="") 
    stream = groq_client(message_history)

    for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)
            await asyncio.sleep(0.05)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()