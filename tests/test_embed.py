import os
from unittest.mock import patch, MagicMock
from embed import load_documents, split_documents, create_embedding_model, create_vectorstore


def test_load_documents(tmp_path):
    """Test that load_documents reads .txt files from a directory and returns Document objects."""
    dummy_file = tmp_path / "example.txt"
    dummy_file.write_text("This is a test.")

    docs = load_documents(str(tmp_path))
    assert len(docs) == 1
    assert "This is a test." in docs[0].page_content


def test_split_documents():
    """Test that split_documents splits long text into multiple chunks of Document."""
    from langchain.schema import Document
    docs = [Document(page_content="A long string " * 100)]

    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(isinstance(c, Document) for c in chunks)


@patch("embed.FAISS")
def test_create_vectorstore(mock_faiss):
    """Test that create_vectorstore builds a FAISS store and saves it locally."""
    from langchain.schema import Document

    mock_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_store

    docs = [Document(page_content="hello world")]
    embeddings = MagicMock()
    out = create_vectorstore(docs, embeddings, "/tmp/fake-path")

    mock_faiss.from_documents.assert_called_once()
    mock_store.save_local.assert_called_with("/tmp/fake-path")
    assert out == mock_store
