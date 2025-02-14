# semantic_store_vectorstore.py
"""
Split the text using a semantic-based splitter (SemanticChunker).
Build and store a vector store (FAISS) from the semantic chunks,
using a cached version if the same PDF file has been processed before.
"""

import os
import hashlib
from typing import List

from load_env import get_api_credentials
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS

from pdf_loader import load_pdf_document


def compute_file_hash(file_path: str) -> str:
    """
    Compute an MD5 hash of the file's content to use as a cache key.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def create_semantic_documents(
    page_texts: List[str], use_standard_deviation: bool = True, threshold: float = 1.25
) -> List[Document]:
    """
    Given a list of page texts, split the text using a semantic-based splitter (SemanticChunker).
    """
    full_text = "\n\n".join(page_texts)

    # Initialize SemanticChunker with OpenAIEmbeddings
    if use_standard_deviation:
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(),
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=threshold,
        )
    else:
        text_splitter = SemanticChunker(OpenAIEmbeddings())

    # Create the document objects.
    documents: List[Document] = text_splitter.create_documents([full_text])

    # Optionally, show progress for document creation if needed (using tqdm for long lists).
    # For now, we assume a single long text so no loop is necessary.
    return documents


def create_or_load_vectorstore(
    documents: List[Document], vectorstore_path: str
) -> FAISS:
    """
    Create a FAISS vector store from the given semantic documents and store it locally.
    If the vector store already exists at the given path, load and return it.
    """
    embeddings = OpenAIEmbeddings()

    if os.path.exists(vectorstore_path):
        # Load with dangerous deserialization enabled (trusting our own cache)
        print(f"Found the cached vector store at {vectorstore_path}. Loading it...")
        vectorstore: FAISS = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print(
            f"Building the vector store for the new document at {vectorstore_path}..."
        )
        texts: List[str] = [document.page_content for document in documents]
        # If processing many texts, you can wrap this loop with tqdm, for example:
        # texts = [text for text in tqdm(texts, desc="Processing document texts")]
        vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)

    return vectorstore


def main() -> None:
    """
    1. Prompt the user for a PDF file path.
    2. Load the PDF document and get the page texts.
    3. Create the semantic document chunks using SemanticChunker.
    4. Build or load a FAISS vector store from the chunks, using a cached version if available.
    """
    get_api_credentials()

    pdf_path: str = input("Enter the path to the PDF file: ").strip()
    if not pdf_path:
        raise ValueError("No file path provided.")

    try:
        # Compute a unique hash for the PDF to use for caching.
        file_hash: str = compute_file_hash(pdf_path)
        cache_dir: str = "./vectorstore_cache"
        os.makedirs(cache_dir, exist_ok=True)
        vectorstore_path: str = os.path.join(cache_dir, file_hash)

        # Load the PDF document (list of page texts).
        page_texts: List[str] = load_pdf_document(pdf_path)

        # Create the semantic documents.
        documents: List[Document] = create_semantic_documents(page_texts)

        # Create or load the vector store from cache.
        _: FAISS = create_or_load_vectorstore(documents, vectorstore_path)

        # Final informational message.
        print("Vector store is ready.")
    except Exception as e:
        print(f"Failed to process PDF document: {e}")


if __name__ == "__main__":
    main()
