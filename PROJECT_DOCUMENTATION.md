# FinSight: Project Documentation

## 1. Project Overview

FinSight is a Python web application that provides on-demand Natural Language Processing (NLP) analysis for financial texts. It uses a custom-trained spaCy model to perform Named Entity Recognition (NER) and a rule-based model (VADER) for sentiment analysis. The application features a dark-themed, modern user interface for submitting text and viewing the interactive analysis.

This project demonstrates the entire lifecycle of an applied NLP project: data analysis, rule-based system creation, model fine-tuning, and application development with Flask.

### Final File Structure

```
finsight/
│
├── app.py
├── train.py
├── training_data.py
│
├── templates/
│   └── index.html
│
└── trained_model_final/
    └── (spaCy model data)
```

---

## 2. The Training Pipeline

The core of this project is a custom-trained NLP model. The training pipeline consists of two files: `training_data.py` and `train.py`.

### `training_data.py` - The "Answer Key"

This file provides the "ground truth" data used to fine-tune our model. It contains a list of examples where we have manually labeled the entities we want the model to learn.

**Code:**
```python
# training_data.py

# This is the format spaCy needs for training:
# ("Text of the sentence", {"entities": [(start_char, end_char, "LABEL")]})
# start_char is the index of the first character of the entity.
# end_char is the index of the first character AFTER the entity.
TRAIN_DATA = [
    ("US stock market valuations at historic highs seen before great depression, dot-com crash.",
     {"entities": [(0, 2, "GPE"), (54, 72, "FIN_EVENT"), (74, 88, "FIN_EVENT")]}),

    ("The US stock market valuation has hit historic highs, with metrics like market-cap-to-GDP exceeding the Great Depression of 1929 and the dot-com crash in 2000.",
     {"entities": [(4, 6, "GPE"), (106, 134, "FIN_EVENT"), (139, 163, "FIN_EVENT")]}),

    ("For context, in 1999, the CAPE hit about 44 before the crash.",
     {"entities": [(18, 22, "DATE"), (53, 64, "FIN_EVENT")]}),

    ("Tech giants like MSFT and IBM also saw gains.",
     {"entities": [(17, 21, "STOCK"), (26, 29, "STOCK")]})
]
```

**Explanation:**
- The `TRAIN_DATA` list holds tuples. Each tuple is one training example.
- The first item in the tuple is the sentence text.
- The second item is a dictionary. The `entities` key holds a list of all the entities in that sentence.
- Each entity is defined by its `start` character, `end` character, and the `LABEL` we want to assign (e.g., `FIN_EVENT`, `STOCK`, `GPE`).
- By providing examples of default entities (`GPE`, `DATE`) alongside our custom ones, we prevent the "catastrophic forgetting" problem and remind the model of what it already knows.

### `train.py` - The Model Trainer

This script handles the process of fine-tuning the pre-trained spaCy model with our custom data.

**Code:**
```python
# train.py (Final Version)
import spacy
import random
from spacy.training.example import Example
from training_data import TRAIN_DATA

model_to_use = "en_core_web_md"
new_model_name = "trained_model_final"
n_iterations = 30

print(f"Loading base model: {model_to_use}")
nlp = spacy.load(model_to_use)
ner = nlp.get_pipe("ner")

# Add the new labels to the NER model's vocabulary
ner.add_label("FIN_EVENT")
ner.add_label("STOCK")

# Get the names of the pipes we want to disable during training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

print("Starting Training...")
with nlp.disable_pipes(*unaffected_pipes):
    for iteration in range(n_iterations):
        print(f"--- Iteration {iteration + 1} of {n_iterations} ---")
        random.shuffle(TRAIN_DATA)
        losses = {}
        
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.35, losses=losses)
            
        print(f"Losses: {losses}")

print(f"
Training complete. Saving final model to ./{new_model_name}")
nlp.to_disk(new_model_name)
```

**Explanation:**
1.  **Load Model:** It loads the base `en_core_web_md` model.
2.  **Add Labels:** It informs the Named Entity Recognition (`ner`) component that we are teaching it two new labels: `FIN_EVENT` and `STOCK`.
3.  **Disable Other Pipes:** It disables all other parts of the NLP pipeline (like the parser and tagger) to make the training focused and efficient.
4.  **Training Loop:** It loops through our `TRAIN_DATA` for 30 iterations. In each iteration, it shuffles the data and shows each example to the model via `nlp.update()`. This function compares the model's prediction to our "correct" answer and slightly adjusts the model's internal weights to be more accurate.
5.  **Save Model:** After the loop finishes, it saves the newly fine-tuned model to a new folder on disk, `trained_model_final`.

---

## 3. The Web Application

The application consists of the Flask backend (`app.py`) and the HTML/CSS/JS frontend (`index.html`).

### `app.py` - The Backend Server

This script uses the Flask framework to create a web server that exposes our NLP pipeline as an API.

**Code:**
```python
# app.py (Final Version)
from flask import Flask, request, jsonify, render_template
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

app = Flask(__name__)

LABEL_COLORS = {
    "PERSON": "#aa9cfc", "ORG": "#7aecec", "GPE": "#feca74",
    "DATE": "#bce784", "FIN_EVENT": "#ff9999", "STOCK": "#ffb3c1",
}

CUSTOM_DESCRIPTIONS = {
    "STOCK": "A stock market ticker symbol.",
    "FIN_EVENT": "A significant financial or market event, like a crash or bubble."
}

print("Loading custom-trained spaCy model...")
# 1. Load our final, custom-trained model
nlp = spacy.load("trained_model_final") 

# 2. Add a rule-based EntityRuler on top of the trained model
patterns = [
    {"label": "STOCK", "pattern": [{"TEXT": "$"}, {"IS_UPPER": True}]}
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
    
    options = {"colors": LABEL_COLORS}
    html = displacy.render(doc, style="ent", options=options)
    
    sentiment_scores = analyzer.polarity_scores(cleaned_text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05: sentiment_label = 'Positive'
    elif compound_score <= -0.05: sentiment_label = 'Negative'
    else: sentiment_label = 'Neutral'
        
    unique_labels = sorted(list(set([ent.label_ for ent in doc.ents])))
    legend = {label: {"description": CUSTOM_DESCRIPTIONS.get(label, spacy.explain(label)), "color": LABEL_COLORS.get(label, "#ddd")} for label in unique_labels}

    return jsonify({
        'html': html, 'legend': legend,
        'sentiment_score': compound_score, 'sentiment_label': sentiment_label
    })

if __name__ == '__main__':
    app.run(debug=True)
```

**Explanation:**
- **Hybrid Model:** The app uses a powerful hybrid approach. It first loads our `trained_model_final`, which understands context. Then, it adds an `EntityRuler` on top to guarantee that it *always* finds tickers that follow a specific pattern (e.g., `$AAPL`), making the system both smart and precise.
- **Preprocessing:** The `preprocess_text` function performs a simple but crucial cleaning step on the user's input.
- **`/analyze` Endpoint:** This is the core of the API. It receives text from the frontend, runs it through the full NLP pipeline (including sentiment analysis with VADER), generates the `displacy` visualization, creates the data for the legend, and sends everything back in a clean JSON format.

### `templates/index.html` - The Frontend

This file contains all the code for the user interface, including the structure (HTML), styling (CSS), and interactivity (JavaScript).

**Code:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FinSight Analyzer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px auto; max-width: 1200px; background-color: #1e1e1e; color: #e0e0e0; }
        textarea { width: 100%; box-sizing: border-box; border: 1px solid #444; border-radius: 5px; padding: 10px; font-size: 1rem; background-color: #2d2d2d; color: #e0e0e0; }
        textarea::placeholder { color: #888; }
        button { background-color: #007bff; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; font-size: 1rem; }
        #analysis-container { display: flex; flex-direction: row; gap: 20px; margin-top: 20px; }
        #entity-vis { flex-grow: 1; border: 1px solid #444; padding: 20px; border-radius: 5px; max-height: 600px; overflow-y: auto; }
        #legend { flex-shrink: 0; width: 300px; }
        .entities { line-height: 2.5; }
        .entity { color: #1e1e1e; font-weight: bold; }
        table { width: 100%; margin-top: 20px; border-collapse: collapse; }
        th, td { border: 1px solid #444; padding: 8px; text-align: left; font-size: 0.9rem; }
        th { background-color: #3a3a3a; }
        #sentiment { font-size: 1.2rem; font-weight: bold; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h1>FinSight 📈</h1>
    <p>This application uses a custom-trained NLP model to identify entities and a rule-based model for sentiment in financial text.</p>
    
    <form id="nlp-form">
        <textarea id="text-input" rows="10" placeholder="Paste an article here..."></textarea>
        <br><br>
        <button type="submit">Analyze Text</button>
    </form>

    <div id="analysis-container">
        <div id="entity-vis"></div>
        <div id="legend"></div>
    </div>
    <div id="sentiment"></div>

    <script>
        document.getElementById('nlp-form').addEventListener('submit', async function(event) {
            event.preventDefault();
            const text = document.getElementById('text-input').value;
            const entityVisDiv = document.getElementById('entity-vis');
            const legendDiv = document.getElementById('legend');
            const sentimentDiv = document.getElementById('sentiment');

            entityVisDiv.innerHTML = 'Analyzing...';
            legendDiv.innerHTML = '';
            sentimentDiv.innerHTML = '';

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text }),
            });

            const data = await response.json();
            
            entityVisDiv.innerHTML = data.html;
            sentimentDiv.innerHTML = `<h3>Sentiment Analysis</h3><p>Overall Sentiment: <strong>${data.sentiment_label}</strong> (Score: ${data.sentiment_score.toFixed(3)})</p>`;

            if (data.legend && Object.keys(data.legend).length > 0) {
                let legendHtml = '<h3>Entity Legend</h3><table><tr><th>Label</th><th>Description</th></tr>';
                for (const label in data.legend) {
                    const color = data.legend[label].color;
                    const description = data.legend[label].description;
                    const coloredLabel = `<span style="background-color:${color}; color: #1e1e1e; padding: 0.2em 0.4em; border-radius: 0.35em; font-weight: bold;">${label}</span>`;
                    legendHtml += `<tr><td>${coloredLabel}</td><td>${description}</td></tr>`;
                }
                legendHtml += '</table>';
                legendDiv.innerHTML = legendHtml;
            }
        });
    </script>

</body>
</html>
```

**Explanation:**
- **CSS:** The `<style>` block defines the dark theme, the side-by-side flexbox layout for the results, and the styling for all components, including the legend table.
- **HTML:** The `<body>` defines the structure, including the main text area and the `div`s that act as placeholders for the analysis results (`entity-vis`, `legend`, `sentiment`).
- **JavaScript:** The `<script>` block handles all user interactivity. It listens for the form submission, uses the `fetch` API to send the text to our Flask backend, and then dynamically builds the HTML for the sentiment score and the color-coded legend table from the JSON data it receives back.
