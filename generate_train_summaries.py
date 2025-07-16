import os.path
import json
import sys
from tqdm import tqdm
import torch
from transformers import GenerationConfig, StoppingCriteria, AutoTokenizer, AutoModelForCausalLM
from blingfire import text_to_sentences

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dataset_names = ["mesh_descriptors", "fos", "search", "same_author", "high_influence_cite", "cite_prediction", "cite_prediction_new"]
total_number_of_files = sum([len(os.listdir(os.path.join(os.path.dirname(__file__), f"data/abstract_data/{d}/abstracts"))) for d in dataset_names])

prompts = ["To summarize, the key findings of our research, stated in one sentence that includes all relevant information, are that",
           "In summary, our research is concerned with",
           "In summary, a comprehensive and detailed conclusive statement would be that",
           "A comprehensive summary for our work would be that",
           "The main takeaway from our work is that"]

stop = [". A", ". B", ". C", ". D", ". E", ". F", ". G", ". H", ". I", ". J", ". K", ". L", ". M",
        ". N", ". O", ". P", ". Q", ". R", ". S", ". T", ". U", ". V", ". W", ". X", ". Y", ". Z", ".<|", ".\n"]

class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, num_sequences=3):
        self.tokenizer = tokenizer
        self.num_sequences = num_sequences
        self.finished_sequences = [False] * num_sequences

    def __call__(self, input_ids, scores, **kwargs):
        for i in range(self.num_sequences):
            if not self.finished_sequences[i]:
                if self.tokenizer.decode(input_ids[i, -2:]).strip()[:3] in stop or self.tokenizer.decode(input_ids[i, -1:]) == stop[-1]:
                    self.finished_sequences[i] = True

        return all(self.finished_sequences)

    def reset(self):
        self.finished_sequences = [False] * self.num_sequences

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.bfloat16)

    generation_config = GenerationConfig(
        max_new_tokens=100,
        num_beams=1,
        no_repeat_ngram_size=5,
        output_scores=False,
        length_penalty=1.0,
        repetition_penalty=1.0,
        top_k=3,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id)
    stopping_criteria = MyStoppingCriteria(tokenizer)
    return model, tokenizer, generation_config, stopping_criteria

def generate_continuations(text, prompts, tokenizer, model, generation_config, stopping_criteria):
    continuations = []

    with torch.no_grad():
        for prompt in prompts:
            input_batch = tokenizer([text + " " + prompt], return_tensors="pt", truncation=True, max_length=1500).to("cuda")

            input_length = input_batch["input_ids"].shape[1]
            stopping_criteria.reset()
            model_output = model.generate(**input_batch, generation_config=generation_config, stopping_criteria=stopping_criteria)

            hypotheses = []
            for output in model_output:
                try:
                    summary_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                    summary_text = summary_text.strip()
                    if summary_text[0] == ":":
                        summary_text = summary_text[1:].strip()
                    if summary_text[0] == "\"":
                        summary_text = summary_text[1:]
                    sentence = text_to_sentences(summary_text).split("\n")[0]
                    if len(sentence) > 20:
                        hypotheses.append(sentence)
                except:
                    continue
            continuations.append((prompt, hypotheses))
    return continuations

def generate_summaries(start_idx, end_idx):
    model, tokenizer, generation_config, stopping_criteria = load_llama_model()

    idx = 0
    for dataset_name in dataset_names:
        bar = None
        for filename in sorted(os.listdir(os.path.join(os.path.dirname(__file__), f"../outputs/abstract_data/{dataset_name}/abstracts"))):
            idx += 1
            if idx < start_idx or idx > end_idx:
                continue

            if os.path.exists(os.path.join(os.path.dirname(__file__), f"../outputs/abstract_data/{dataset_name}/summaries/{filename[:-4]}.json")):
                bar = None
                continue

            if bar is None:
                bar = tqdm(total=total_number_of_files, initial=idx, desc=f"Generating ({dataset_name})...", smoothing=0.05, file=sys.stdout)

            with open(os.path.join(os.path.dirname(__file__), f"../outputs/abstract_data/{dataset_name}/abstracts/{filename}"), encoding="utf-8") as f:
                title, abstract = f.read().split("\n")[:2]
            text = f"{title}{'.' if len(title) > 0 and title[-1] != '.' else ''} {abstract}"

            continuations = generate_continuations(text, prompts, tokenizer, model, generation_config, stopping_criteria)
            if continuations is None:
                continue

            output_file = os.path.join(os.path.dirname(__file__), f"../outputs/abstract_data/{dataset_name}/summaries/{filename[:-4]}.json")
            with open(output_file, "w") as f:
                json.dump(continuations, f)
            bar.update(1)

if __name__ == '__main__':
    generate_summaries(0, 400_000)