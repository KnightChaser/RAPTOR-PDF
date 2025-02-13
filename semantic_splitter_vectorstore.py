# semantic_store_vectorstore.py
"""
Split the text using a semantic-based splitter (SemanticChunker).
Build and store a vector store (FAISS) from the semantic chunks.
"""
from typing import List

from load_env import get_api_credentials
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS


from pdf_loader import load_pdf_document


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

    # Create the document objects
    documents: List[Document] = text_splitter.create_documents([full_text])

    if documents:
        print("First document chunk:")
        print(documents[0].page_content)
        print(
            f"{type(documents)} => {type(documents[0])} => {type(documents[0].page_content)}"
        )
        print("=" * 50)

    return documents


def create_and_store_vectorstore(
    documents: List[Document], vectorstore_path: str = "./vectorstore_index"
) -> FAISS:
    """
    Create a FAISS vectorstore from the given semantic documetns and store it locally.
    And return the created vectorstore object as well.
    """
    texts: List[str] = [document.page_content for document in documents]
    embeddings = OpenAIEmbeddings()
    vectorstore: FAISS = FAISS.from_texts(texts=texts, embedding=embeddings)

    vectorstore.save_local(vectorstore_path)
    print(f"Vector store saved to {vectorstore_path}")

    return vectorstore


def main() -> None:
    """
    1. Prompt the user for a PDF file path.
    2. Load the PDF document(pdf_loader.py) and get the page texts.
    3. Create the semantic document chunks using SemanticChunker.
    4. Build and store a FAISS vector store from the chunks.
    """
    print("Loading environment variables...")
    get_api_credentials()

    pdf_path: str = input("Enter the path to the PDF file: ")
    if not pdf_path:
        raise ValueError("No file path provided.")

    try:
        page_texts: List[str] = load_pdf_document(pdf_path)
        print(
            f"Loaded {len(page_texts)} pages from {pdf_path}."
            f"Total content length: {sum(len(text) for text in page_texts)}"
        )

        # Create the semantic documents
        documents: List[Document] = create_semantic_documents(page_texts)
        print(f"Created {len(documents)} semantic documents.")

        # Create and store the vector store
        vectorstore: FAISS = create_and_store_vectorstore(documents)

        print("Vector store created and saved successfully.")
        print(f"Vector store object: {vectorstore}")
    except Exception as e:
        print(f"Failed to load PDF document: {e}")


if __name__ == "__main__":
    main()
