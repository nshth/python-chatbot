import chainlit as cl
from utils import files_handler, build_rag_chain, llm
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

@cl.on_chat_start
async def start():
    cl.user_session.set("rag_chain", None)
    cl.user_session.set("message_history", [
        SystemMessage(content="You are a helpful assistant.")
    ])
    await cl.Message(
        content="Hi! You can chat with me directly, or upload a PDF/text file to ask questions about it."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    uploaded_file = []
    for el in message.elements:
        if isinstance(el, cl.File):
            uploaded_file.append(el)

    msg = cl.Message(content="")

    if uploaded_file:
        processing = cl.Message(content="Processing uploaded files...")
        await processing.send()

        files_handler(uploaded_file)
        cl.user_session.set("rag_chain", build_rag_chain())

        rag_chain = cl.user_session.get("rag_chain")

        async for event in rag_chain.astream_events(
            {"input": message.content},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    await msg.stream_token(token)

    else:
        history = cl.user_session.get("message_history")
        history.append(HumanMessage(content=message.content))

        async for chunk in llm.astream(history):
            if chunk.content:
                await msg.stream_token(chunk.content)

        history.append(AIMessage(content=msg.content))
        cl.user_session.set("message_history", history)

    await msg.update()