from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os
from pydantic import BaseModel, Field
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.schema import Document  # ✅ Import Document class

def perform_ocr(client, file_path: str, model: str = "mistral-ocr-latest"):
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        exit()

    uploaded_file = client.files.upload(
        file={"file_name": file_path, "content": file_bytes},
        purpose="ocr"
    )

    file_signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    file_url = file_signed_url.url

    response = client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": file_url},
        include_image_base64=True
    )

    return response


def get_markdown_from_ocr(response):
    resp_dict = response.model_dump()
    pages = resp_dict.get("pages", [])
    # Return list of tuples (page_number, markdown_text)
    all_md = [(i+1, p.get("markdown", "")) for i, p in enumerate(pages) if p.get("markdown")]
    return all_md

# ---------------- Markdown Splitting ----------------
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def split_md(doc_dict):
    """
    doc_dict = {"source": filename, "page": page_number, "text": md_text}
    Creates chunks from markdown splits AND expanded chunks by concatenating with up to next 3 chunks.
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=True)
    md_splits = splitter.split_text(doc_dict["text"])  # list of Document objects

    base_chunks = []
    seen = set()
    for doc in md_splits:
        text = doc.page_content.strip()
        if text and text not in seen:
            base_chunks.append(Document(
                page_content=text,
                metadata={"source": doc_dict["source"], "page": doc_dict["page"]}
            ))
            seen.add(text)

    # Expand chunks by concatenating up to next 3 chunks
    expanded_chunks = []
    for i in range(len(base_chunks)):
        # Start with current chunk
        combined_text = base_chunks[i].page_content
        expanded_chunks.append(Document(
            page_content=combined_text,
            metadata=base_chunks[i].metadata
        ))

        # Add combinations with next 1, 2, 3 chunks
        for j in range(1, 3):
            if i + j < len(base_chunks):
                combined_text += " " + base_chunks[i + j].page_content
                expanded_chunks.append(Document(
                    page_content=combined_text,
                    metadata=base_chunks[i].metadata  # keep metadata of starting chunk
                ))

    return expanded_chunks




# ---------------- PDF Extraction ----------------
def extract_pdf_text_mistral(client, pdfs):
    docs = []
    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        response = perform_ocr(client, pdf_path)
        page_md_list = get_markdown_from_ocr(response)  # list of (page_number, markdown)
        for page, md_text in page_md_list:
            docs.append({
                "source": pdf,
                "page": page,
                "text": md_text
            })
    return docs


def extract_pdf_text(pdfs):
    docs = []
    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        docs.extend(PyPDFLoader(pdf_path).load())
    return docs

# ---------------- Text Chunks ----------------
def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=0, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

# ---------------- Vectorstore ----------------
def get_vectorstore(pdfs, from_session_state=False, persist_directory=None):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    load_dotenv()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    '''if from_session_state and persist_directory and os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        return vectordb'''

    docs_with_metadata = extract_pdf_text_mistral(client, pdfs)
    chunks = []
    for doc in docs_with_metadata:
        chunks.extend(split_md(doc))

    print(f"📑 Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {i} (Page {chunk.metadata['page']}) ---\n{chunk.page_content}\n{'-'*50}\n")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb, docs_with_metadata








'''
def get_vectorstore(pdfs, from_session_state=False):
    """
    Create or retrieve a vectorstore from PDF documents

    Parameters:
    - pdfs (list): List of PDF documents
    - from_session_state (bool, optional): Flag indicating whether to load from session state. Defaults to False

    Returns:
    - vectordb or None: The created or retrieved vectorstore. Returns None if loading from session state and the database does not exist
    """
    load_dotenv()
    
    # Use HuggingFace embeddings instead of GoogleGenerativeAI
    
    # Initialize the model normally first
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    # Wrap in LangChain HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(client=sbert_model)

    if from_session_state and os.path.exists("Vector_DB - Documents"):
        # Retrieve vectorstore from existing one
        vectordb = Chroma(persist_directory="Vector_DB - Documents", embedding_function=embedding)
        return vectordb

    elif not from_session_state:
        docs = extract_pdf_text(pdfs)
        chunks = get_text_chunks(docs)
        print(f"Number of chunks: {len(chunks)}")
        print(f"First chunk: {chunks[0].page_content}")
        # Create vectorstore from chunks and save it to the folder Vector_DB - Documents
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory="Vector_DB - Documents"
        )
        return vectordb

    return None
'''