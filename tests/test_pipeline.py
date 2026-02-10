from nlp.pipeline import _classify_sentiment
from nlp.preprocessing import preprocess_text


class TestClassifySentiment:
    def test_positive_sentiment(self):
        assert _classify_sentiment(0.05) == "Positive"
        assert _classify_sentiment(0.8) == "Positive"

    def test_negative_sentiment(self):
        assert _classify_sentiment(-0.05) == "Negative"
        assert _classify_sentiment(-0.9) == "Negative"

    def test_neutral_sentiment(self):
        assert _classify_sentiment(0.0) == "Neutral"
        assert _classify_sentiment(0.04) == "Neutral"
        assert _classify_sentiment(-0.04) == "Neutral"


class TestPreprocessText:
    def test_normalizes_whitespace(self):
        assert preprocess_text("  hello   world  ") == "hello world"

    def test_preserves_normal_text(self):
        assert preprocess_text("hello world") == "hello world"

    def test_handles_newlines_and_tabs(self):
        assert preprocess_text("hello\n\tworld") == "hello world"

    def test_empty_string(self):
        assert preprocess_text("") == ""
