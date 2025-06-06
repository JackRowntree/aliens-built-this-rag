import os
import glob
import logging
from typing import List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_documents(doc_dir: str) -> List[Document]:
    """
    Load .md and .txt files from a directory into LangChain Documents.
    """
    pattern = os.path.join(doc_dir, "*.md")
    files = glob.glob(pattern) + glob.glob(os.path.join(doc_dir, "*.txt"))
    documents = []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append(Document(
            page_content=text,
            metadata={"source": os.path.basename(filepath)}
        ))

    logger.info(f"Loaded {len(documents)} documents from {doc_dir}")
    return documents


def split_documents(documents: List[Document], chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:
    """
    Chunk each document using overlapping text splits.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Load embedding model via LangChain wrapper.
    """
    logger.info(f"Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vectorstore(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Optional[str] = "data/vectordb",
    collection_name: str = "conspiracy_docs"
) -> Chroma:
    """
    Create and persist a Chroma vector store (or return in-memory if persist_directory is None).
    """
    logger.info(f"Building vectorstore with {len(chunks)} chunks")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    if persist_directory:
        vectordb.persist()
        logger.info(f"Persisted vectorstore to: {persist_directory}")
    else:
        logger.warning("Running vectorstore in memory — will not persist between runs.")

    return vectordb


def build_vectorstore_from_docs(
    doc_dir: str = "docs/",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_directory: Optional[str] = "data/vectordb"
) -> Chroma:
    """
    End-to-end pipeline: Load → Chunk → Embed → Vectorstore
    """
    documents = load_documents(doc_dir)
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    embeddings = create_embedding_model(embedding_model_name)
    return create_vectorstore(chunks, embeddings, persist_directory=persist_directory)


if __name__ == "__main__":
    build_vectorstore_from_docs()
