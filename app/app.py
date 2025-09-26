import streamlit as st
import os
import shutil
import time
import gc
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat

# --------------------------
# Utility to safely delete folders (Windows-friendly)
# --------------------------
def delete_folder(folder_path, retries=5, delay=0.5):
    if os.path.exists(folder_path):
        for _ in range(retries):
            try:
                shutil.rmtree(folder_path)
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(delay)
        else:
            st.warning(f"Could not delete {folder_path}. Make sure no program is using it.")

# --------------------------
# Main ChatApp
# --------------------------
class ChatApp:
    def __init__(self):
        # Ensure docs folder exists
        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Streamlit config
        st.set_page_config(page_title="Insurance Chatbot 📚")
        st.title("Insurance Chatbot 📚")

        # ✅ Only clear old data once per app session
        if "app_initialized" not in st.session_state:
            self.clear_old_data()
            st.session_state.app_initialized = True

        # Initialize session state
        initialize_session_state_variables(st)

        # Initialize vectordb/session variables if they don't exist
        if "vectordb" not in st.session_state:
            st.session_state.vectordb = None
        if "docs" not in st.session_state:
            st.session_state.docs = []

    # --------------------------
    # Clear old PDFs and Vector DB
    # --------------------------
    def clear_old_data(self):
        # Release vectordb if exists
        if "vectordb" in st.session_state and st.session_state.vectordb:
            try:
                del st.session_state.vectordb._client
            except AttributeError:
                pass
            st.session_state.vectordb = None
            gc.collect()

        # Delete vector DB folder
        delete_folder("Vector_DB - Documents")

        # Delete all uploaded PDFs
        docs_folder = "docs"
        if os.path.exists(docs_folder):
            for file_name in os.listdir(docs_folder):
                file_path = os.path.join(docs_folder, file_name)
                try:
                    os.remove(file_path)
                except PermissionError:
                    pass

        # Clear session state keys
        for key in ["chat_history", "uploaded_pdfs", "processed_documents", "vectordb", "previous_upload_docs_length", "docs"]:
            st.session_state.pop(key, None)

    # --------------------------
    # Main run loop
    # --------------------------
    def run(self):
        # Sidebar for uploading documents
        with st.sidebar:
            st.subheader("Your documents")
            upload_docs = os.listdir("docs")
            st.session_state.processed_documents = upload_docs
            if upload_docs:
                st.text(", ".join(upload_docs))
            else:
                st.info("No documents uploaded yet.")

            # File uploader
            st.subheader("Upload PDF documents")
            pdf_docs = st.file_uploader(
                "Select a PDF document and click on 'Process'",
                type=['pdf'],
                accept_multiple_files=True
            )
            if pdf_docs:
                # Save PDFs
                save_docs_to_vectordb(pdf_docs, upload_docs)

        # Refresh folder after potential uploads
        upload_docs = os.listdir("docs")
        st.session_state.processed_documents = upload_docs

        # --------------------------
        # Initialize or update vectorstore
        # --------------------------
        if upload_docs:
            if st.session_state.vectordb is None or len(upload_docs) > st.session_state.get("previous_upload_docs_length", 0):
                vectordb, docs = get_vectorstore(
                    upload_docs,
                    from_session_state=False,
                    persist_directory=None  # in-memory
                )
                st.session_state.vectordb = vectordb
                st.session_state.docs = docs
                st.session_state.previous_upload_docs_length = len(upload_docs)

            # --------------------------
            # Run chat
            # --------------------------
            st.session_state.chat_history = chat(
                st.session_state.chat_history,
                st.session_state.vectordb,
                st.session_state.docs  # pass docs for BM25/hybrid
            )
        else:
            st.info(
                "Upload a PDF file to chat with it. "
                "You can keep uploading files to chat with, and if you leave, you won't need to upload these files again."
            )


# --------------------------
# Run the app
# --------------------------
if __name__ == "__main__":
    app = ChatApp()
    app.run()
