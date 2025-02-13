# pdf_loader.py
"""
Load PDF files from a user input(file path) and analyze the token amount distribution within the file.
"""

import os
import sys
import warnings
from typing import List

import tiktoken
from langchain_community.document_loaders import PDFPlumberLoader


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Calculate the number of tokens in a given string using the specified encoding scheme.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(string)

    return len(tokens)


def load_pdf_document(pdf_path: str) -> List[str]:
    """
    Load a PDF document and return a list of its page contents.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()
    page_texts = [document.page_content for document in documents]

    return page_texts


def main() -> None:
    """
    Main function to prompt the user for a PDF file path,
    and analyze the token amount distribution within the file.
    """
    pdf_path = input("Enter the path to the PDF file: ")
    if not pdf_path:
        raise ValueError("No file path provided.")

    try:
        page_texts = load_pdf_document(pdf_path)
        print(
            f"Loaded {len(page_texts)} pages from {pdf_path}."
            f"Total content length: {sum(len(text) for text in page_texts)}"
        )

        # Print the result
        print(f"{'Page #':<10}{'Token Count':<15}")
        print("-" * 25)

        token_counts = [num_tokens_from_string(text) for text in page_texts]
        for i, count in enumerate(token_counts, start=1):
            print(f"{i:<10}{count:<15}")
    except Exception as e:
        print(f"Failed to load PDF document: {e}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
