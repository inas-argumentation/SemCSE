import json
import os
from blingfire import text_to_sentences

def relative_path(path):
    return os.path.join(os.path.dirname(__file__), path)

def load_abstract_with_summaries(dataset_name, filename, load_sentences=False, tokenizer=None):
    with open(os.path.join(relative_path(f"data/abstract_data/{dataset_name}/abstracts"), f"{filename}.txt"), "r", encoding="utf-8") as f:
        elements = f.read().split("\n")
        title = elements[0]
        abstract = ' '.join(elements[1:])
        text = f"{title}{'.' if len(title) > 0 and title[-1].isalpha() else ''} {abstract}" if tokenizer is None else f"{title} {tokenizer.sep_token} {abstract}"
    if True:
        with open(os.path.join(relative_path(f"data/abstract_data/{dataset_name}/summaries"), f"{filename}.json"), "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            summaries = [z.strip() for x, y in data for z in y if len(z) > 4]
    if load_sentences:
        sentences = [str(x) for x in text_to_sentences(text).split("\n") if len(str(x)) > 20]
        return text, summaries, title, sentences

    return text, summaries