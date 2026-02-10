import logging
from typing import Any

import spacy
from spacy import displacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from nlp.preprocessing import preprocess_text

logger = logging.getLogger(__name__)

LABEL_COLORS: dict[str, str] = {
    "PERSON": "#aa9cfc",
    "ORG": "#7aecec",
    "GPE": "#feca74",
    "DATE": "#bce784",
    "FIN_EVENT": "#ff9999",
    "STOCK": "#ffb3c1",
    "CARDINAL": "#e4e7d2",
    "MONEY": "#e4e7d2",
    "PERCENT": "#e4e7d2",
}

CUSTOM_DESCRIPTIONS: dict[str, str] = {
    "STOCK": "A stock market ticker symbol.",
    "FIN_EVENT": "A significant financial or market event, like a crash or bubble.",
}

nlp = None
analyzer = None


def load_model(model_path: str) -> None:
    global nlp, analyzer

    logger.info("Loading custom-trained spaCy model from '%s'...", model_path)
    nlp = spacy.load(model_path)

    patterns = [
        # STOCK — $TICKER format
        {"label": "STOCK", "pattern": [{"TEXT": "$"}, {"IS_UPPER": True}]},
        # STOCK — known tickers standing alone
        {"label": "STOCK", "pattern": [{"TEXT": {"IN": [
            "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "TSLA", "META", "NVDA",
            "NFLX", "JPM", "BA", "LMT", "IBM", "AMD", "INTC", "DIS", "V",
            "WMT", "KO", "PEP", "NKE", "PYPL", "SQ", "UBER", "SNAP"
        ]}}]},
        # FIN_EVENT — keyword phrases
        {"label": "FIN_EVENT", "pattern": [{"LOWER": {"IN": ["financial", "economic", "banking", "market"]}}, {"LOWER": "crisis"}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": {"IN": ["stock", "market", "flash", "crypto"]}}, {"LOWER": "crash"}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": {"IN": ["housing", "tech", "dot-com", "crypto", "asset"]}}, {"LOWER": "bubble"}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": "bubble"}, {"LOWER": "burst"}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": "black"}, {"LOWER": {"IN": ["monday", "tuesday", "wednesday", "thursday", "friday"]}}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": "great"}, {"LOWER": {"IN": ["depression", "recession"]}}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": "bear"}, {"LOWER": "market"}]},
        {"label": "FIN_EVENT", "pattern": [{"LOWER": "bull"}, {"LOWER": "market"}]},
    ]
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns(patterns)

    analyzer = SentimentIntensityAnalyzer()

    # Extend VADER's lexicon with finance-specific terms
    # Scores range from -4 (most negative) to +4 (most positive)
    analyzer.lexicon.update({
        "crash": -3.5,
        "crashed": -3.5,
        "crashing": -3.5,
        "tanking": -3.0,
        "tanked": -3.0,
        "plunge": -3.0,
        "plunged": -3.0,
        "plummeted": -3.0,
        "selloff": -2.5,
        "sell-off": -2.5,
        "downturn": -2.5,
        "recession": -3.0,
        "depression": -3.5,
        "bearish": -2.0,
        "headwinds": -1.5,
        "nervousness": -2.0,
        "tariff": -1.5,
        "tariffs": -1.5,
        "overvalued": -1.5,
        "expensive": -1.0,
        "decline": -2.0,
        "declined": -2.0,
        "dropping": -2.0,
        "dropped": -2.0,
        "falling": -2.0,
        "fell": -2.0,
        "slump": -2.5,
        "slumped": -2.5,
        "correction": -1.5,
        "volatile": -1.5,
        "volatility": -1.5,
        "rally": 2.5,
        "bullish": 2.0,
        "surge": 2.5,
        "surged": 2.5,
        "breakout": 2.0,
        "outperform": 2.0,
        "upgrade": 1.5,
        "upgraded": 1.5,
        "undervalued": 1.5,
        "upbeat": 2.0,
    })

    logger.info("Model and pipeline ready.")


def _correct_entities(doc: Any) -> list:
    corrected = []
    for ent in doc.ents:
        if ent.label_ == "STOCK":
            if ent.text.isupper() or ent.text.startswith("$"):
                corrected.append(ent)
        else:
            corrected.append(ent)
    return corrected


def _classify_sentiment(compound_score: float) -> str:
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_text(raw_text: str) -> dict:
    cleaned_text = preprocess_text(raw_text)
    doc = nlp(cleaned_text)

    doc.ents = _correct_entities(doc)

    html = displacy.render(doc, style="ent", options={"colors": LABEL_COLORS})

    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    compound_score = sentiment_scores["compound"]
    sentiment_label = _classify_sentiment(compound_score)

    unique_labels = sorted({ent.label_ for ent in doc.ents})
    legend = {
        label: {
            "description": CUSTOM_DESCRIPTIONS.get(label, spacy.explain(label)),
            "color": LABEL_COLORS.get(label, "#ddd"),
        }
        for label in unique_labels
    }

    return {
        "html": html,
        "legend": legend,
        "sentiment_score": compound_score,
        "sentiment_label": sentiment_label,
    }
