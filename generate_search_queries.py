import os
import sys
import json
from tqdm import tqdm
from llama_cpp import Llama
from load_data import relative_path

output_path = relative_path("data/semantic_similarity_dataset_queries.json")

def generate_search_queries(model_path = "bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF"):
    with open(relative_path("data/semantic_similarity_dataset.json"), "r") as f:
        abstract_data = json.loads(f.read())

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_data = json.loads(f.read())
    else:
        existing_data = {}

    parameters_map = {
        "Mistral": {
            "temperature": 0.15,
            "top_k": 50,
            "top_p": 1,
            "min_p": 0.01
        }
    }

    model_params = parameters_map["Mistral"]
    llm = Llama.from_pretrained(
        repo_id=model_path,
        filename="*Q8_0.gguf",
        n_gpu_layers=-1,
        n_ctx=6000,
        verbose=False)

    for idx, data in enumerate(tqdm(abstract_data, file=sys.stdout, desc="Creating Search Queries")):
        if str(idx) in existing_data and len(existing_data[str(idx)]) > 10:
            continue

        title = data[0].strip()
        abstract = data[1].strip()

        if title and title[-1].isalpha():
            title += "."

        combined_text = f"{title} {abstract}"

        prompt_text = (
            "You are given a scientific paper title and abstract. "
            "Your task is to create a search query that a user could hypothetically have entered into an academic search engine. "
            "The query you create shall be a query that fits the given paper, meaning that the search engine should ideally return this paper as search result for the query. "
            "The output must be valid JSON containing exactly one key 'query' with a string containing the query.\n"
            "For example: {\"query\": \"Your search query.\"}\n"
            "Here are title and abstract of the article:\n"
            f"{combined_text}\n"
        )

        messages = [
            {"role": "system", "content": "You are a research assistant tasked with creating data for training and evaluating a scientific search engine."},
            {"role": "user", "content": prompt_text}
        ]

        for i in range(10):
            response = llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1000,
                **model_params)

            raw_content = response["choices"][0]["message"]["content"]
            try:
                parsed_json = json.loads(raw_content)
                query = parsed_json.get("query", "")
                if len(query) > 10:
                    existing_data[str(idx)] = query
                    break

            except json.JSONDecodeError:
                continue

        with open(output_path, "w") as f:
            json.dump(existing_data, f)


if __name__ == '__main__':
    generate_search_queries()