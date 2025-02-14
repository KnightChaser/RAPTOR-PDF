# qna_service.py
"""
Module for running the RAG QnA service.
This module uses a vectorstore (FAISS) to retrieve relevant document contexts and answer user queries
via a retrieval-augmented generation (RAG) chain.
"""

import warnings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_documents(docs):
    """
    Format a list of documents into a single string.

    This function now checks if each document is a dict (e.g. returned by the retriever)
    or an object with a 'page_content' attribute, and extracts the text accordingly.
    """
    formatted = []
    for doc in docs:
        if isinstance(doc, dict):
            # Try to get 'page_content' or fallback to str(doc)
            content = doc.get("page_content", str(doc))
        else:
            # Assume doc has a page_content attribute
            content = getattr(doc, "page_content", str(doc))
        formatted.append(f"<document>{content}</document>")
    return "\n\n".join(formatted)


def run_rag_chain(vectorstore):
    """
    Build a RAG chain that retrieves context from the vectorstore and generates an answer.

    Args:
        vectorstore: The FAISS vectorstore containing embedded texts.

    Returns:
        A chain that can be invoked with a question.
    """
    # Create a retriever from the vectorstore.
    retriever = vectorstore.as_retriever()

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
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def run_qna_service(vectorstore):
    """
    Run a QnA loop where the user can ask questions about the document and receive answers.

    Args:
        vectorstore: The FAISS vectorstore to use for retrieval.
    """
    print("\nQnA service is ready. Type 'exit' to quit.")
    rag_chain = run_rag_chain(vectorstore)
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Exiting QnA service. See you next time!")
            break

        # Invoke the chain with the user's question.
        answer_chunk = []
        print("Answer: ")
        for answer in rag_chain.stream(question):
            answer_chunk.append(answer)
            print(answer, end="")
        print("\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(
        "This module is intended to be imported and used via run_qna_service(vectorstore)."
    )
