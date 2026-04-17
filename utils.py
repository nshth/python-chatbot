import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv("./.env")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
)

def files_handler(files: list) -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in files:
        path = file.path
        name = file.name

        # load
        if name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding="utf-8")

        docs = loader.load()

        # tag each chunk with the source filename
        for doc in docs:
            doc.metadata["source"] = name
        
        # chunk -> embed -> add to vector store
        chunks = splitter.split_documents(docs)
        vector_store.add_documents(chunks)
        print(f"Stored {len(chunks)} chunks from {name}")

def build_rag_chain():
    system_prompt = """You are a helpful assistant. Answer using the context below.
        If the answer isn't in the context, say you don't find any related topic in the doc and then say the answer.

        Context:
        {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    combine_docs_chain = create_stuff_documents_chain(llm, prompt) #dump everything into one prompt
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return rag_chain