import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from datasets import load_dataset
from load_data import relative_path
import torch
import torch.nn.functional as F
from blingfire import text_to_sentences
import json
from tqdm import tqdm
import traceback

P_AT_K = 1

with open(relative_path("data/semantic_similarity_dataset.json"), "r") as f:
    data = json.loads(f.read())

with open(relative_path("data/semantic_similarity_dataset_queries.json"), "r") as f:
    queries = json.loads(f.read())

def add_eos(model, input_examples):
  input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
  return input_examples

class SciRepEvalModel:
    def __init__(self, model, tokenizer, output_processing=False, load_dataset_specific_model=False):
        self.model = model
        if output_processing != "encode":
            self.model.eval()
        self.tokenizer = tokenizer
        self.output_processing = output_processing
        self._task_id = None
        self.load_dataset_specific_model = load_dataset_specific_model

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    def __call__(self, batch, batch_ids=None):
        batch = [batch] if type(batch) == str else batch
        if self.output_processing == "NV":
            try:
                inputs = add_eos(self.model, batch)
                embeddings = self.model.encode(inputs, batch_size=4, show_progress_bar=False, normalize_embeddings=True)
            except Exception as e:
                print(e)
                print("Exception done.")
                quit()
            return torch.tensor(embeddings, device="cuda")
        if self.output_processing == "encode":
            try:
                return self.model(batch)
            except:
                print(traceback.format_exc())
                quit()

        input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                   return_tensors="pt", return_token_type_ids=False, max_length=512)
        input_ids.to('cuda')
        output = self.model(**input_ids)

        if self.output_processing == "last_hidden":
            output = output.last_hidden_state[:, 0, :]
        return output

def batched_similarity(embeddings1, embeddings2, metric, batch_size=100):
    n1 = embeddings1.size(0)
    n2 = embeddings2.size(0)
    similarities = torch.zeros((n1, n2), device=embeddings1.device)

    for i in range(0, n1, batch_size):
        for j in range(0, n2, batch_size):
            batch1 = embeddings1[i:i + batch_size]
            batch2 = embeddings2[j:j + batch_size]

            if metric == "cosine":
                sim = F.cosine_similarity(batch1.unsqueeze(1), batch2.unsqueeze(0), dim=-1)
            elif metric == "euclidean":
                sim = -torch.cdist(batch1, batch2)
            else:
                raise Exception("Metric not known.")
            similarities[i:i + batch_size, j:j + batch_size] = sim

    return similarities

def get_embeddings_batched(model, texts, batch_size=8):
    embeddings_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting embeddings", file=sys.stdout):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model(batch_texts)
            embeddings_list.append(batch_embeddings)
    return torch.cat(embeddings_list, dim=0)

def evaluate_query_abstract_match(model, metric):
    titles = [item[0] for item in data]
    abstracts = [item[1] for item in data]
    queries_list = [queries[str(x)] for x in range(len(queries))]
    samples = [(f"{x}. {y}" if x[-1].isalpha() else f"{x} {y}") for x, y in zip(titles, abstracts)]

    sample_embeddings = get_embeddings_batched(model, samples)
    query_embeddings = get_embeddings_batched(model, queries_list)

    similarities = batched_similarity(sample_embeddings, query_embeddings, metric)
    n = len(titles)

    rankings = (-similarities).argsort(dim=1)
    correct_indices = torch.arange(n, device=rankings.device)
    ranks = (rankings == correct_indices.unsqueeze(1)).nonzero()[:, 1]

    return float(ranks.float().mean().item()) + 1

def evaluate_title_abstract_match(model, metric):
    titles = [item[0] for item in data]
    abstracts = [item[1] for item in data]

    title_embeddings = get_embeddings_batched(model, titles)
    abstract_embeddings = get_embeddings_batched(model, abstracts)

    similarities = batched_similarity(title_embeddings, abstract_embeddings, metric)

    rankings = (-similarities).argsort(dim=1)
    correct_indices = torch.arange(len(titles), device=rankings.device)
    ranks = (rankings == correct_indices.unsqueeze(1)).nonzero()[:, 1]

    return float(ranks.float().mean().item()) + 1


def evaluate_abstract_split_match(model, metric):
    abstracts = [item[1] for item in data]

    first_halves = []
    second_halves = []

    for abstract in abstracts:
        sentences = text_to_sentences(abstract).split("\n")
        if not sentences or len(sentences) <= 1:
            continue

        total_chars = sum(len(s) for s in sentences)
        current_chars = 0
        split_idx = 0

        for i, sentence in enumerate(sentences):
            if current_chars + len(sentence) > total_chars / 2 and i > 0:
                break
            current_chars += len(sentence)
            split_idx = i

        first_half = " ".join(sentences[:split_idx + 1])
        second_half = " ".join(sentences[split_idx + 1:])

        first_halves.append(first_half)
        second_halves.append(second_half)

    first_half_embeddings = get_embeddings_batched(model, first_halves)
    second_half_embeddings = get_embeddings_batched(model, second_halves)

    similarities = batched_similarity(first_half_embeddings, second_half_embeddings, metric)

    rankings = (-similarities).argsort(dim=1)
    correct_indices = torch.arange(len(first_halves), device=rankings.device)
    ranks = (rankings == correct_indices.unsqueeze(1)).nonzero()[:, 1]

    return float(ranks.float().mean().item()) + 1

def evaluate_MAG_clustering(model, metric, k=5):
    dataset = load_dataset("allenai/scirepeval", "scidocs_mag_mesh")
    dataset_labels = load_dataset("allenai/scirepeval_test", "scidocs_mag")

    raw_data = {}
    for x in tqdm(dataset["evaluation"], desc="processing_dataset", file=sys.stdout):
        raw_data[x["doc_id"]] = (x["title"], x["abstract"])

    data = {"train": [], "test": []}

    # Process data and labels
    for split in ["train", "test"]:
        for x in tqdm(dataset_labels[split], desc=f"Processing {split} labels", file=sys.stdout):
            if x["paper_id"] in raw_data:
                data[split].append((*raw_data[x["paper_id"]], x["label"]))

    train_texts = [f"{title}. {abstract}" for title, abstract, _ in data["train"]]
    test_texts = [f"{title}. {abstract}" for title, abstract, _ in data["test"]]
    train_labels = [label for _, _, label in data["train"]]
    test_labels = [label for _, _, label in data["test"]]

    unique_labels = sorted(set(train_labels + test_labels))
    num_labels = len(unique_labels)
    print(f"Number of unique labels: {num_labels}")

    print("Computing train embeddings...")
    train_embeddings = get_embeddings_batched(model, train_texts)
    print("Computing test embeddings...")
    test_embeddings = get_embeddings_batched(model, test_texts)

    print("Computing similarities...")
    similarities = batched_similarity(test_embeddings, train_embeddings, metric)

    _, top_k_indices = torch.topk(similarities, k=k, dim=1)

    accuracies = []
    for i, test_label in enumerate(test_labels):
        neighbor_labels = [train_labels[idx] for idx in top_k_indices[i]]
        accuracy = sum(1 for label in neighbor_labels if label == test_label) / k
        accuracies.append(accuracy)

    mean_accuracy = sum(accuracies) / len(accuracies)

    if True:
        import numpy as np
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        print("Creating t-SNE visualization...")

        # Convert test embeddings to numpy and compute t-SNE
        test_embeddings_np = test_embeddings.cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        test_embeddings_2d = tsne.fit_transform(test_embeddings_np)

        # Create scatter plot
        plt.figure(figsize=(12, 8))

        # Create a color map for the labels
        unique_labels_list = list(set(test_labels))
        color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels_list)))

        # Plot each point
        for label in unique_labels_list:
            mask = np.array(test_labels) == label
            plt.scatter(
                test_embeddings_2d[mask, 0],
                test_embeddings_2d[mask, 1],
                label=f'Label {label}',
                alpha=0.6)

        plt.title('t-SNE visualization of test samples')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('tsne_visualization.png')
        plt.close()

    return mean_accuracy

def evaluate_model_semantics(model, tokenizer, output_processing=None, metric="euclidean"):
    model = SciRepEvalModel(model, tokenizer, output_processing=output_processing)
    query_score = evaluate_query_abstract_match(model, metric)
    print(f"Query-abstract match: {query_score:.3f}")
    title_abstract_score = evaluate_title_abstract_match(model, metric)
    print(f"Title-abstract match: {title_abstract_score:.3f}")
    abstract_split_score = evaluate_abstract_split_match(model, metric)
    print(f"Abstract-split match: {abstract_split_score:.3f}")
    mag_clustering_score = evaluate_MAG_clustering(model, metric)
    print(f"MAG clustering score: {mag_clustering_score:.3f}")
    print(f"Scores: {title_abstract_score:.2f} & {abstract_split_score:.2f} & {query_score:.2f} & {mag_clustering_score:.3f}")