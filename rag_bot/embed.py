"""
Embedding and vector store utilities for converting raw text documents into FAISS-indexed embeddings.
"""
import os
import logging
from pathlib import Path
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader

logger = logging.getLogger(__name__)


def load_documents(doc_dir: str) -> List[Document]:
    """
    Load all `.txt` files from a directory as LangChain Document objects.

    Args:
        doc_dir (str): Path to a directory containing plain-text files.

    Returns:
        List[Document]: A list of loaded LangChain documents.
    """
    docs = []
    for path in Path(doc_dir).glob("*.txt"):
        loader = TextLoader(str(path))
        docs.extend(loader.load())
    logger.info(f"Loaded {len(docs)} documents from {doc_dir}")
    return docs


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller overlapping chunks using recursive character splitting.

    Args:
        docs (List[Document]): A list of LangChain documents to split.
        chunk_size (int): The target number of characters per chunk.
        chunk_overlap (int): The number of characters of overlap between chunks.

    Returns:
        List[Document]: A list of chunked LangChain documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def create_vectorstore(chunks: List[Document], embeddings, persist_directory: str) -> FAISS:
    """
    Create a FAISS vector store from a list of document chunks and save it to disk.

    Args:
        chunks (List[Document]): Document chunks to embed and index.
        embeddings: A LangChain-compatible embedding model.
        persist_directory (str): Directory path to store the FAISS index.

    Returns:
        FAISS: A FAISS vector store containing the embedded document chunks.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_directory)
    logger.info(f"Saved FAISS vector store to {persist_directory}")
    return vectorstore


def build_vectorstore_from_docs(
    doc_dir: str,
    persist_directory: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> FAISS:
    """
    Full pipeline: load documents, split them, embed them, and persist a FAISS index to disk.

    Args:
        doc_dir (str): Path to directory of input `.txt` files.
        persist_directory (str): Output path for saving the FAISS index.
        chunk_size (int): Character length for each text chunk.
        chunk_overlap (int): Character overlap between text chunks.
        embedding_model_name (str): Name of the embedding model to use.

    Returns:
        FAISS: A persisted FAISS vector store ready for similarity search.
    """
    documents = load_documents(doc_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    embeddings = HuggingFaceEmbeddings(embedding_model_name)
    return create_vectorstore(chunks, embeddings, persist_directory)
