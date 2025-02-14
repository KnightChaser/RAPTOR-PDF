# qna_service.py
"""
Module for running the RAG QnA service.
This module uses a vectorstore (FAISS) to retrieve relevant document contexts and answer user queries
via a retrieval-augmented generation (RAG) chain.
"""

import warnings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_documents(docs):
    """
    Format a list of documents into a single string.

    This function checks if each document is a dict (e.g. returned by the retriever)
    or an object with a 'page_content' attribute, and extracts the text accordingly.
    """
    formatted = []
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get("page_content", str(doc))
        else:
            content = getattr(doc, "page_content", str(doc))
        formatted.append(f"<document>{content}</document>")
    return "\n\n".join(formatted)


def create_rag_chain(vectorstore):
    """
    Build a RAG chain that retrieves context from the vectorstore, prints which documents
    were retrieved in an organized manner, and generates an answer.

    Args:
        vectorstore: The FAISS vectorstore containing embedded texts.

    Returns:
        A chain that can be invoked with a question.
    """
    # Create a retriever from the vectorstore.
    retriever = vectorstore.as_retriever()

    def retrieve_and_print(question):
        """
        Retrieve documents based on the question, print them, and return a formatted string.
        """
        # Retrieve documents for the given question.
        # (Assuming your retriever uses get_relevant_documents method)
        docs = retriever.get_relevant_documents(question)

        # Print the retrieved documents in an organized manner.
        print("\n--- Retrieved Documents ---")
        for idx, doc in enumerate(docs, start=1):
            if isinstance(doc, dict):
                content = doc.get("page_content", str(doc))
            else:
                content = getattr(doc, "page_content", str(doc))
            print(f"Document {idx}:\n{content}\n{'-' * 30}")
        print("---------------------------\n")

        # Return the formatted documents for use in the chain.
        return format_documents(docs)

    # Define the prompt template.
    prompt_template = """
    You are an AI assistant specializing in Question-Answering (QA) tasks within a Retrieval-Augmented Generation (RAG) system. 
    You are given PDF documents. Your primary mission is to answer questions based on provided context.
    Ensure your response is concise and directly addresses the question without any additional narration.

    ###
    Your final answer should be written concisely (but include important numerical values, technical terms, jargon, and names).
    ###
    Remember:
    - It's crucial to base your answer solely on the **PROVIDED CONTEXT**. 
    - DO NOT use any external knowledge or information not present in the given materials.
    ###
    # Here is the user's QUESTION that you should answer:
    {question}
    
    # Here is the CONTEXT that you should use to answer the question:
    {context}
    
    [Note]
    - Answer should be written in formal and technical English.
    
    # Your final ANSWER to the user's QUESTION:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Initialize the language model.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Build the RAG chain.
    rag_chain = (
        {"context": retrieve_and_print, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ask_raptor_rag(vectorstore: FAISS, question: str) -> None:
    """
    Ask a question to the RAPTOR RAG chain and print the answer on the console.

    Args:
        question (str): The question to ask the RAG chain.
    """
    if not question:
        print("Error: Please provide a question.")
        return

    rag_chain = create_rag_chain(vectorstore)
    answer_chunk = []
    print("Answer: ", end="")
    for answer in rag_chain.stream(question):
        answer_chunk.append(answer)
        print(answer, end="")
    print("\n")


def run_qna_service(vectorstore):
    """
    Run a QnA loop where the user can ask questions about the document and receive answers.

    Args:
        vectorstore: The FAISS vectorstore to use for retrieval.
    """
    print("\nQnA service is ready. Type 'exit' to quit.")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting QnA service. See you next time!")
            break

        # Fire the question!!! Yes, we are ready to answer.
        ask_raptor_rag(vectorstore, question)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(
        "This module is intended to be imported and used via run_qna_service(vectorstore)."
    )
