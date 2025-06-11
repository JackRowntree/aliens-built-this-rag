import pytest
from unittest.mock import MagicMock, patch

from rag import load_vectorstore, get_llm, get_qa_chain, ask_question


@patch("rag.FAISS.load_local")
@patch("rag.HuggingFaceEmbeddings")
def test_load_vectorstore(mock_embeddings, mock_load_local):
    """Tests vectorstore loading with mocked dependencies."""
    mock_vectorstore = MagicMock()
    mock_load_local.return_value = mock_vectorstore

    vs = load_vectorstore("some/path")
    assert vs == mock_vectorstore
    mock_load_local.assert_called_once()
    mock_embeddings.assert_called_once()


@patch("rag.OpenAI")
def test_get_llm(mock_openai):
    """Tests that an OpenAI LLM is initialized properly."""
    mock_llm = MagicMock()
    mock_openai.return_value = mock_llm

    llm = get_llm()
    assert llm == mock_llm
    mock_openai.assert_called_once()


def test_get_qa_chain_returns_chain():
    """Tests QA chain assembly using a mocked vector store."""
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = "mock-retriever"

    with patch("rag.get_llm") as mock_get_llm, \
         patch("rag.RetrievalQA.from_chain_type") as mock_chain_type:
        mock_get_llm.return_value = "mock-llm"
        mock_chain = MagicMock()
        mock_chain_type.return_value = mock_chain

        result = get_qa_chain(mock_vectorstore)
        assert result == mock_chain
        mock_chain_type.assert_called_once()


def test_ask_question_returns_string():
    """Tests that ask_question correctly processes output from the chain."""
    mock_chain = MagicMock()
    mock_chain.return_value = {"result": "ancient astronauts built the pyramids"}

    answer = ask_question(mock_chain, "Who built the pyramids?")
    assert answer == "ancient astronauts built the pyramids"
