# main.py
"""
Main module for the RAPTOR PDF processing service.

Workflow:
1. A PDF file path is provided externally.
2. The PDF is loaded and analyzed (with token count details printed).
3. The content is semantically split into multiple chunks.
4. Each chunk is recursively processed via the RAPTOR summarization chain.
5. Both original chunks and hierarchical summaries are aggregated.
6. The combined texts are embedded and stored in a FAISS vector database.
7. The vectorstore object is returned for further retrieval-augmented QnA.
"""

import os
import sys
import warnings
from typing import List

from tqdm import tqdm

from qna_service import run_qna_service
from load_env import get_api_credentials
from pdf_loader import load_pdf_document, num_tokens_from_string
from semantic_splitter_vectorstore import create_semantic_documents
from raptor_recursive import recursive_embed_cluster_summarize
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_vectorstore_from_pdf(pdf_file_path: str) -> FAISS:
    """
    Full processing pipeline:
      - Load PDF.
      - Perform semantic splitting.
      - Run recursive RAPTOR summarization.
      - Aggregate original text chunks and all summaries.
      - Build and store a FAISS vectorstore.

    Args:
        pdf_file_path (str): The absolute path to the PDF file.

    Returns:
        FAISS: The vectorstore object containing embeddings from the aggregated texts.
    """
    warnings.filterwarnings("ignore")
    get_api_credentials()

    if not os.path.exists(pdf_file_path):
        print("Invalid file path. Exiting.")
        sys.exit(1)

    # Step 1: Load the PDF file.
    print("Loading PDF file...")
    page_texts: List[str] = load_pdf_document(pdf_file_path)
    print(f"Loaded {len(page_texts)} pages from PDF.")

    # Calculate total tokens and display progress.
    total_tokens = sum(num_tokens_from_string(text) for text in page_texts)
    print(f"Total tokens in PDF: {total_tokens}")

    # Step 2: Semantic splitting.
    print("Performing semantic splitting on PDF content...")
    documents = create_semantic_documents(
        page_texts, use_standard_deviation=True, threshold=1.25
    )
    print(f"Semantic splitting complete: {len(documents)} document chunks created.")

    # Convert Document objects to plain text chunks.
    text_chunks: List[str] = [doc.page_content for doc in documents]

    # Step 3: Recursive RAPTOR summarization.
    print("Starting RAPTOR recursive summarization...")
    results = recursive_embed_cluster_summarize(text_chunks, level=1, n_levels=3)
    highest_level = max(results.keys())
    _final_clusters, final_summary = results[highest_level]
    print(f"RAPTOR recursive summarization complete. Final level: {highest_level}")
    print(f"Number of final summarized clusters: {len(final_summary)}")

    # Step 4: Aggregate original chunks and summaries.
    print("Aggregating texts for vector store creation...")
    all_texts: List[str] = text_chunks.copy()
    # Append summaries from each recursion level.
    for level in sorted(results.keys()):
        df_summary = results[level][1]
        for summary in tqdm(
            df_summary["summaries"].tolist(),
            desc=f"Processing summaries for level {level}",
        ):
            all_texts.append(summary)

    # Step 5: Build and store the FAISS vector store.
    print("Building vector store from aggregated texts...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=all_texts, embedding=embeddings)
    vectorstore_path = "./vectorstore_index"
    vectorstore.save_local(vectorstore_path)
    print(f"Vector store saved to {vectorstore_path}.")

    return vectorstore


def main() -> None:
    # Determine PDF file path from command-line arguments.
    if len(sys.argv) > 1:
        pdf_file_path = sys.argv[1]
    else:
        print("Usage: python main.py /path/to/pdf_file")
        sys.exit(1)

    vectorstore = build_vectorstore_from_pdf(pdf_file_path)
    print("RAPTOR service processing complete. Ready for QnA retrieval!")

    # Hand off the vectorstore to the QnA service.
    run_qna_service(vectorstore)


if __name__ == "__main__":
    main()
