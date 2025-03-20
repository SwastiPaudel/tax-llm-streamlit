import streamlit as st
import os
import dotenv
import uuid

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai  import ChatGoogleGenerativeAI

from rag_methods import (
    load_doc_to_db,
    stream_llm_rag_response,
    initialize_vector_db,
)

from db_service import get_training_file_names

dotenv.load_dotenv()

if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        # "openai/o1-mini",
        "openai/gpt-4o",
        # "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
        "google/gemini-2.0-flash"
    ]
else:
    MODELS = ["openai/gpt-4o"]


st.set_page_config(
    page_title="Tax Helper",
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;"><i> Ask your Tax related questions? </i></h2>""")


# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
]


openai_api_key = os.getenv("OPENAI_API_KEY")
st.session_state.openai_api_key = openai_api_key
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
st.session_state.anthropic_api_key = anthropic_api_key
gemini_api_key = os.getenv("GEMINI_API_KEY")
st.session_state.gemini_api_key = gemini_api_key

# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key
missing_anthropic = anthropic_api_key == "" or anthropic_api_key is None
missing_gemini = gemini_api_key == "" or gemini_api_key is None

is_vector_db_loaded = ("vector_db" in st.session_state and st.session_state.vector_db is not None)

if not is_vector_db_loaded:
    st.session_state.vector_db = initialize_vector_db()
    is_vector_db_loaded = True

if "rag_sources" not in st.session_state:
    print("Loading training files...")
    st.session_state.rag_sources = []
    st.session_state.rag_sources.extend(get_training_file_names())

    print(st.session_state.rag_sources)


if missing_openai and missing_anthropic and ("AZ_OPENAI_API_KEY" not in os.environ):
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")

else:
    # Sidebar
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            elif "gemini" in model and not missing_gemini:
                models.append(model)

        st.selectbox(
            "🤖 Select a Model",
            options=models,
            key="model",
        )

        cols0 = st.columns(2)
        with cols0[0]:
            st.toggle(
                "Use RAG", 
                value=is_vector_db_loaded, 
                key="use_rag", 
                disabled=not is_vector_db_loaded,
            )

        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        st.header("Tax Document Sources:")
            
        # File upload input for RAG with documents
        st.file_uploader(
            "📄 Upload a document", 
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            on_change=load_doc_to_db,
            key="rag_docs",
        )

        # # URL input for RAG with websites
        # st.text_input(
        #     "🌐 Introduce a URL",
        #     placeholder="https://example.com",
        #     on_change=load_url_to_db,
        #     key="rag_url",
        # )

        print(is_vector_db_loaded)

        with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
            st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

    
    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_api_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.2,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_api_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.2,
            streaming=True,
        )
    elif model_provider == "google":
        llm_stream = ChatGoogleGenerativeAI(
            api_key=gemini_api_key,
            model="gemini-2.0-flash",
            temperature=0.2,
            streaming=True
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
            with st.spinner(text="Getting your answer please wait..."):
                stream_llm_rag_response(llm_stream, messages)


with st.sidebar:
    st.divider()

    
