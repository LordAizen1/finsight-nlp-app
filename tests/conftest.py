from unittest.mock import patch, MagicMock
import pytest

from nlp.pipeline import _classify_sentiment


@pytest.fixture()
def mock_nlp():
    """Mock the spaCy model and VADER analyzer so tests don't need model artifacts."""
    mock_doc = MagicMock()
    mock_doc.ents = []

    mock_model = MagicMock()
    mock_model.return_value = mock_doc

    mock_ruler = MagicMock()
    mock_model.add_pipe.return_value = mock_ruler

    mock_analyzer = MagicMock()
    mock_analyzer.polarity_scores.return_value = {
        "compound": 0.5, "pos": 0.3, "neg": 0.0, "neu": 0.7,
    }

    with patch("nlp.pipeline.spacy") as mock_spacy, \
         patch("nlp.pipeline.SentimentIntensityAnalyzer", return_value=mock_analyzer), \
         patch("nlp.pipeline.displacy") as mock_displacy:
        mock_spacy.load.return_value = mock_model
        mock_displacy.render.return_value = "<div>mocked html</div>"
        mock_spacy.explain.return_value = "An entity"

        from nlp.pipeline import load_model
        load_model("fake_model")

        yield {
            "model": mock_model,
            "analyzer": mock_analyzer,
            "displacy": mock_displacy,
            "doc": mock_doc,
        }


@pytest.fixture()
def client(mock_nlp):
    """Flask test client with mocked NLP pipeline."""
    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
