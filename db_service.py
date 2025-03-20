import streamlit as st


def get_training_file_names():
    """
    Retrieves the list of training file names stored in the 'training_files_metadata' collection.

    :return: List of stored file names
    """
    chroma_client = st.session_state.vector_db._client
    collection = chroma_client.get_or_create_collection("training_files_metadata")

    # Fetch stored filenames
    docs = collection.get()

    # Extract file names from metadata
    file_names = [metadata["file_name"] for metadata in docs["metadatas"] if "file_name" in metadata]

    if not file_names:
        print("No training files found in ChromaDB.")
        return []
    else:
        print(f"Found {len(file_names)} training files in ChromaDB.")
        print(file_names)
        return file_names


def store_training_files(file_name):
    """
    Stores the names of files used for training in a dedicated ChromaDB collection.

    :param file_name:
    """
    chroma_client = st.session_state.vector_db._client
    collection = chroma_client.get_or_create_collection("training_files_metadata")

    # Store filenames in ChromaDB

    collection.add(
        ids=[file_name],  # Use filename as unique ID
        embeddings=[[0.0] * 1536],  # Placeholder embedding
        metadatas=[{"file_name": file_name}]
    )

    print(f"Stored {len(file_name)} as training file in ChromaDB.")