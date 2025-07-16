import os
import sys
from load_data import relative_path, load_abstract_with_summaries
from model import load_model, save_model
from settings import *
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

random.seed(0)

dataset_names = ["mesh_descriptors", "fos", "search", "same_author", "high_influence_cite", "cite_prediction_new"]
filenames = {dataset_name: os.listdir(relative_path(f"data/abstract_data/{dataset_name}/summaries")) for dataset_name in dataset_names}
filenames = {x: y[:int(len(y)/1)] for x, y in filenames.items()}

dataset_sizes = {name: len(files) for name, files in filenames.items()}
total_size = sum(float(np.sqrt(size)) for size in dataset_sizes.values())
dataset_probs = {name: float(np.sqrt(size)) / total_size for name, size in dataset_sizes.items()}
available_files = {name: list(files) for name, files in filenames.items()}
print(sum(size for size in dataset_sizes.values()))

def reset_data():
    global filenames, eval_samples, eval_sentences, eval_abstracts

    eval_samples = None
    eval_sentences = None

    for dataset_name in filenames:
        random.shuffle(filenames[dataset_name])
reset_data()

# Compute similarities between two sets of embeddings in a batched way
def batched_similarity(embeddings1, embeddings2, batch_size=100):
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

# Split off first samples from each dataset to create eval set
def create_eval_dataset():
    global eval_samples, eval_sentences, eval_abstracts

    eval_samples = []
    for dataset in filenames:
        eval_samples.extend([(dataset, x) for x in filenames[dataset][:150]])
        filenames[dataset] = filenames[dataset][150:]

    eval_sentences = []
    eval_abstracts = []
    for i, (dataset, filename) in enumerate(eval_samples):
        text, summaries = load_abstract_with_summaries(dataset, filename[:-5])
        if len(summaries) >= 2:
            eval_sentences.append((dataset, random.sample(summaries, 2)))
            eval_abstracts.append((dataset, text))

# Get a new random sample for training
def get_random_sample(dataset_name, include_abstract_sentences, tokenizer=None):
    global available_files
    if not available_files[dataset_name]:
        available_files[dataset_name] = list(filenames[dataset_name])
        random.shuffle(available_files[dataset_name])
    filename = available_files[dataset_name].pop()
    return load_abstract_with_summaries(dataset_name, filename[:-5], load_sentences=include_abstract_sentences, tokenizer=tokenizer)

# Create a batch of random summaries/sentences for training
def create_batch(batch_size=SENTENCE_BATCH_SIZE, dataset_name=None, include_abstract_sentences=True, tokenizer=None):
    global available_files
    batch = []

    if not dataset_name:
        n_datasets = 5
        batch_dataset_names = [random.choices(list(dataset_probs.keys()), weights=list(dataset_probs.values()))[0] for _ in range(n_datasets)]


    while len(batch) < batch_size:
        current_dataset = dataset_name if dataset_name else random.choice(batch_dataset_names)

        if include_abstract_sentences:
            abstract, summaries, title, sentences = get_random_sample(current_dataset, True, tokenizer)
            if len(summaries) < 2: continue

            # Randomly select either two summaries, a summary and the title, or a summary and a random abstract sentence
            p = random.random()
            if p < 0.5:
                selected_summaries = random.sample(summaries, 2)
                batch.append([*selected_summaries, abstract])
            elif p < 0.65:
                selected_summary = random.choice(summaries)
                batch.append([selected_summary, title, abstract])
            else:
                selected_summary = random.choice(summaries)
                if len(sentences) < 1:
                    continue
                selected_sentence = random.choice(sentences)
                batch.append([selected_summary, selected_sentence, abstract])
        else:
            abstract, summaries = get_random_sample(current_dataset, False)
            if len(summaries) < 2: continue
            selected_summaries = random.sample(summaries, 2)
            batch.append(tuple(selected_summaries))

    return batch

def evaluate_embedding_model(model, tokenizer):
    model.eval()

    if not eval_sentences:
        create_eval_dataset()

    all_summaries = [sentence for _, pair in eval_sentences for sentence in pair]
    all_abstracts = [x[1] for x in eval_abstracts]

    with torch.no_grad():
        # Predict the embeddings of all summaries in the eval set
        batch_size = SENTENCE_BATCH_SIZE
        summary_embeddings = []
        for i in range(0, len(all_summaries), batch_size):
            batch = all_summaries[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=510, return_tensors="pt").to("cuda")
            batch_embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            summary_embeddings.append(batch_embeddings)

        # Predict the embeddings of all abstracts in the eval set
        abstract_embeddings = []
        for i in range(0, len(all_abstracts), ABSTRACT_BATCH_SIZE):
            batch = all_abstracts[i:i + ABSTRACT_BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=510, return_tensors="pt").to("cuda")
            batch_embeddings = model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0]
            abstract_embeddings.append(batch_embeddings)

        all_summary_embeddings = torch.cat(summary_embeddings, dim=0)
        all_abstract_embeddings = torch.cat(abstract_embeddings, dim=0)

        # Calculate matching performance for summary embeddings
        summary_similarities = batched_similarity(all_summary_embeddings, all_summary_embeddings)
        summary_similarities.fill_diagonal_(-torch.inf)
        arg_sorted = torch.argsort(summary_similarities, dim=-1, descending=True)
        rank_scores = []
        for i in range(all_summary_embeddings.shape[0]//2):
            correct_rank = (arg_sorted[i*2] == i*2+1).nonzero(as_tuple=True)[0].item()
            rank_scores.append(correct_rank)

            correct_rank = (arg_sorted[i*2+1] == i*2).nonzero(as_tuple=True)[0].item()
            rank_scores.append(correct_rank)
        average_summary_rank = np.mean(rank_scores)
        print(f"Average summary match rank: {average_summary_rank:.3f}")

        # Calculate matching performance for average summary embedding and abstract embedding
        avg_summary_embeddings = (all_summary_embeddings[0::2] + all_summary_embeddings[1::2]) / 2
        similarities = batched_similarity(avg_summary_embeddings, all_abstract_embeddings)
        arg_sorted = torch.argsort(similarities, dim=-1, descending=True)
        rank_scores = []
        for i in range(avg_summary_embeddings.shape[0]):
            correct_rank = (arg_sorted[i] == i).nonzero(as_tuple=True)[0].item()
            rank_scores.append(correct_rank)
        average_abstract_rank = np.mean(rank_scores)
        print(f"Average abstract match rank: {average_abstract_rank:.3f}")
        return -(average_abstract_rank + average_summary_rank)

def train_embedding_model(load_if_available=False):
    model, tokenizer = load_model(load_if_available)
    model = model.to("cuda")

    max_val_score = evaluate_embedding_model(model, tokenizer)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    epochs_without_improvement = -5
    for epoch in range(1, 200):
        model.train()

        for batch_idx in tqdm(range(1000), desc=f"Training (epoch {epoch})", file=sys.stdout):
            batch_sentences = create_batch(include_abstract_sentences=True, tokenizer=None)

            h_1 = [x[0] for x in batch_sentences]
            h_2 = [x[1] for x in batch_sentences]

            batch_1 = tokenizer(h_1, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            batch_2 = tokenizer(h_2, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")

            embeddings_1 = model(**batch_1, output_hidden_states=True).hidden_states[-1][:, 0]
            embeddings_2 = model(**batch_2, output_hidden_states=True).hidden_states[-1][:, 0]

            if metric == "cosine": # Used for ablation on using cosine similarity
                temperature = 0.07
                embeddings_1_norm = F.normalize(embeddings_1, p=2, dim=-1)
                embeddings_2_norm = F.normalize(embeddings_2, p=2, dim=-1)

                sim_matrix = torch.matmul(embeddings_1_norm, embeddings_2_norm.t())

                logits = sim_matrix / temperature

                labels = torch.arange(logits.size(0), device=logits.device)
                contrastive_loss = F.cross_entropy(logits, labels)

                current_contrastive_loss_scaled = contrastive_loss
                norm_penalty = (1 / 250.0) * torch.norm(embeddings_1, dim=-1).mean()

                loss = current_contrastive_loss_scaled + norm_penalty
                loss.backward()
            if metric == "euclidean": # The triplet loss formulation
                distances = torch.cdist(embeddings_1, embeddings_2, p=2)

                positive_distances = distances.diag().unsqueeze(-1).repeat(1, SENTENCE_BATCH_SIZE)
                combined_distances = torch.cat([positive_distances.unsqueeze(-1), distances.unsqueeze(-1)], dim=-1)

                scores = torch.relu(combined_distances[:, :, 0] - combined_distances[:, :, 1] + 1)
                off_diagonal_mask = ~torch.eye(scores.size(0), dtype=torch.bool, device="cuda")
                off_diagonal_entries = scores[off_diagonal_mask]

                loss = off_diagonal_entries.mean()
                loss += 1 / 250 * torch.norm(embeddings_1, dim=-1).mean()

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epochs_without_improvement += 1
        val_score = evaluate_embedding_model(model, tokenizer)
        if val_score > max_val_score:
            max_val_score = val_score

            save_model(model)
            print("New best! Model saved.")
            epochs_without_improvement = min(epochs_without_improvement, 0)

        if epochs_without_improvement >= 15:
            break
