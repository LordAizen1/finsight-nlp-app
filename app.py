from flask import Flask, request, jsonify, render_template
import spacy
from spacy import displacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

app = Flask(__name__)

LABEL_COLORS = {
    "PERSON": "#aa9cfc", "ORG": "#7aecec", "GPE": "#feca74",
    "DATE": "#bce784", "FIN_EVENT": "#ff9999", "STOCK":
"#ffb3c1",
    "CARDINAL": "#e4e7d2", "MONEY": "#e4e7d2", "PERCENT":
"#e4e7d2",
}

CUSTOM_DESCRIPTIONS = {
    "STOCK": "A stock market ticker symbol.",
    "FIN_EVENT": "A significant financial or market event, like a crash or bubble."
}

print("Loading custom-trained spaCy model...")
nlp = spacy.load("trained_model_final")
patterns = [
    {"label": "STOCK", "pattern": [{"TEXT": "$"}, {"IS_UPPER"
: True}]}
]

ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
ruler.add_patterns(patterns)
analyzer = SentimentIntensityAnalyzer()
print("Model and pipeline ready.")

def preprocess_text(text):
    return " ".join(text.split())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    json_data = request.get_json()
    raw_text = json_data['text']
    cleaned_text = preprocess_text(raw_text)
    doc = nlp(cleaned_text)

    corrected_ents = []
    for ent in doc.ents:
        # If the model predicted STOCK...
        if ent.label_ == "STOCK":
            # ...only keep it if the text is all uppercase OR starts with '$'.
            if ent.text.isupper() or ent.text.startswith('$'
):
                corrected_ents.append(ent)
            # Otherwise, we discard the label (e.g., for "Nvidia").
        else:
            # For all other labels (PERSON, ORG, etc.), we keep them.
            corrected_ents.append(ent)

    # Overwrite the document's entities with our corrected list
    doc.ents = corrected_ents

    options = {"colors": LABEL_COLORS}
    html = displacy.render(doc, style="ent", options=options)

    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment_label = 'Positive'
    elif compound_score <= -0.05:
        sentiment_label = 'Negative'
    else:
        sentiment_label = 'Neutral'

    unique_labels = sorted(list(set([ent.label_ for ent in
doc.ents])))
    legend = {label: {"description": CUSTOM_DESCRIPTIONS.get(label, spacy.explain(label)),
"color": LABEL_COLORS.get(label, "#ddd")} for label in
unique_labels}

    return jsonify({
        'html': html,
        'legend': legend,
        'sentiment_score': compound_score,
        'sentiment_label': sentiment_label
    })

if __name__ == '__main__':
    app.run(debug=True)