import chainlit as cl
from utils import files_handler, build_rag_chain

@cl.on_chat_start
async def start():
    rag_chain = build_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)

    files = await cl.AskFileMessage(
        content="Upload a PDF or text file to begin!",
        accept=["text/plain", "application/pdf"],
        max_size_mb=50,
        max_files=3,
        timeout=180,
    ).send()

    if files:
        msg = cl.Message(content="Processing files...")
        await msg.send()
        files_handler(files)          # chunk + store in Chroma
        await cl.Message(content="Files processed! Ask me anything.").send()
    else:
        await cl.Message(content="No files uploaded. You can still ask questions.").send()


@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    msg = cl.Message(content="")
    await msg.send()

    ###
    async for event in rag_chain.astream_events(
        {"input": message.content},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                await msg.stream_token(token)

    await msg.update()