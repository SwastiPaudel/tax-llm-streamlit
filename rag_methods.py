import dotenv
from time import time
import streamlit as st
from pydantic.v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from typing import List

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

from db_service import store_training_files

dotenv.load_dotenv(override=True)

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 50

CHROMA_PATH = "vector_db"

# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        store_training_files(doc_file.name)
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")

                    # finally:
                    #     os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(
                f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.",
                icon="✅")


@st.dialog("Are you sure you want to clear the database? This action cannot be undone.")
def clear_db():
    if st.button("Yes, clear the database"):
        if "vector_db" in st.session_state:
            client = st.session_state.vector_db._client
            collections = client.list_collections()

            # Delete each collection
            for collection in collections:
                try:
                    client.delete_collection(name=collection.name)
                    print(f"Deleted collection: {collection.name}")
                except Exception as e:
                    print(f"Error deleting collection {collection.name}: {e}")

            print("All collections have been deleted.")
            st.session_state.rag_sources = []



def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL *{url}* loaded successfully.", icon="✅")

            else:
                st.error("Maximum number of documents reached (10).")


def initialize_vector_db():
    openai_api_key = os.getenv("OPENAI_API_KEY")

    embedding_func = OpenAIEmbeddings(api_key=openai_api_key)

    # Initialize Chroma vector database
    vector_db = Chroma(
        embedding_function=embedding_func,
        persist_directory=CHROMA_PATH
    )

    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))

    while len(collection_names) > 30:  # Adjust if needed
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db


def load_docs_to_db(docs):
    db = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH,
    )

    db.persist()


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db()

    load_docs_to_db(document_chunks)
    st.session_state.vector_db.add_documents(document_chunks)


class AnswerWithSources(TypedDict):
    """An answer to the question, with sources."""

    answer: Annotated[str, "Answer to the question"]
    sources: Annotated[
        List[str],
        ...,
        "List of sources used to answer the question",
    ]


# --- Retrieval Augmented Generation (RAG) Phase ---
def format_docs(docs):
    docs = "\n\n".join(doc.page_content for doc in docs)
    return docs


def _get_context_retriever_chain(vector_db, llm, messages):
    """Creates a context retriever chain that generates search queries based on conversation history."""
    retriever = vector_db.as_retriever()

    # Document processing prompt
    response_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful tax assistant. Answer the user's query based on the retrieved documents.
        If you don't know the answer, say so. Sources should be formatted [Page number, Topic Title, Document name] but 
        not in the answers. Also, please always provide the sources used to answer the question. Don't answer without source"""),
        ("user", "Context information: {context}"),
        ("user", "Question: {input}")
    ])

    # Chain to generate answer with sources
    rag_chain_from_docs = (
            {
                "input": lambda x: x["input"],  # input query
                "context": lambda x: format_docs(x["context"]),
            }
            | response_prompt
            | llm.with_structured_output(AnswerWithSources)
    )

    # Retrieve documents based on the latest message
    retrieve_docs = (lambda x: x["input"]) | retriever

    # Complete chain
    chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
        answer=rag_chain_from_docs
    )

    return chain


def get_conversational_rag_chain(llm, messages):
    """Creates a conversational RAG chain that incorporates conversation history."""
    return _get_context_retriever_chain(st.session_state.vector_db, llm, messages)


def extract_answer(data):
    """Extracts the 'answer' content from the response JSON."""
    return data.get('answer', {}).get('answer', 'Answer not found')


def get_sources(data):
    return data.get('answer', {}).get('sources', 'sources not found')


def get_context(data):
    return data.get('context', 'context not found')


def get_context_as_list(docs):
    docs_list = [doc.page_content for doc in docs]
    return docs_list


# Simpler approach if you already have the data as a Python dictionary
def stream_llm_rag_response(llm_stream, messages):
    """Streams a RAG response with sources."""
    conversation_rag_chain = get_conversational_rag_chain(llm_stream, messages)

    # Get the response as stream
    # response = conversation_rag_chain.pick('answer').stream({
    #     "input": messages[-1].content,
    #     "messages": messages[:-1]  # Previous messages for context
    # })

    response = conversation_rag_chain.invoke({
        "input": messages[-1].content,
        "messages": messages[:-1]  # Previous messages for context
    })

    with open("response.txt", "a") as file:
        print(response, file=file)

    # Initial setup for response display
    response_text = extract_answer(response)
    sources = get_sources(response)
    context = get_context(response)

    st.write(
        response_text
    )

    with open("context.txt", "a") as file:
        print(get_context_as_list(context), file=file)

    st.markdown(f"##### Sources")

    if not len(sources) < 1:
        for source in sources:
            st.markdown(f"- **{source}**")

    return response_text, get_context_as_list(context), sources


def initialize_app():
    is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)

    if not is_vector_db_loaded:
        st.session_state.vector_db = initialize_vector_db()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.openai_api_key = openai_api_key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    st.session_state.anthropic_api_key = anthropic_api_key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    st.session_state.gemini_api_key = gemini_api_key

    print('Initializing app')

    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
