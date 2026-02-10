import spacy
import random
from spacy.training.example import Example
from training_data import TRAIN_DATA

model_to_use = "en_core_web_md"
new_model_name = "trained_model_final"
n_iterations = 100

print(f"Loading base model: {model_to_use}")
nlp = spacy.load(model_to_use)
ner = nlp.get_pipe("ner")

# Add the new labels to the NER model's vocabulary
ner.add_label("FIN_EVENT")
ner.add_label("STOCK")

# Get the names of the pipes we want to disable during training
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe
not in pipe_exceptions]

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

print(f"\nTraining complete. Saving final model to ./{new_model_name}")
nlp.to_disk(new_model_name)