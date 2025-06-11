"""
This module handles the retrieval and question-answering (RAG) workflow.
"""
import logging
from typing import Optional

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI  # You can switch this to HuggingFacePipeline or others
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def load_vectorstore(persist_directory: str, model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    """
    Loads the FAISS vector store from disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.load_local(persist_directory, embeddings)
    logger.info(f"Loaded FAISS vector store from {persist_directory}")
    return vectorstore


def get_llm(model_name: Optional[str] = None) -> OpenAI:
    """
    Returns an LLM instance (default: OpenAI).
    """
    llm = OpenAI(model_name=model_name or "gpt-3.5-turbo", temperature=0)
    logger.info("Initialized LLM")
    return llm


def get_prompt_template() -> PromptTemplate:
    """
    Returns the prompt template used for RAG-style QA.
    """
    template = """
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.

    Context:
    {context}

    Question:
    {question}
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])


def get_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Creates a RetrievalQA chain with the vector store and LLM.
    """
    retriever = vectorstore.as_retriever()
    llm = get_llm()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    logger.info("QA chain initialized")
    return qa_chain


def ask_question(qa_chain: RetrievalQA, question: str) -> str:
    """
    Asks a question and returns the generated answer.
    """
    logger.info(f"Asking question: {question}")
    result = qa_chain({"query": question})
    answer = result["result"]
    logger.info(f"Answer: {answer}")
    return answer
