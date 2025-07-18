from settings import set_save_name, set_encoder_checkpoint, metric

def train_SemCSE():
    set_save_name("SemCSE")
    set_encoder_checkpoint("KISTI-AI/Scideberta-full")

    # Train model
    from train_embedding_model import train_embedding_model
    train_embedding_model(False)

    # Evaluate on semantic benchmark
    from model import load_model
    model, tokenizer = load_model(True)
    model.eval()

    from semantic_benchmark import evaluate_model_semantics
    evaluate_model_semantics(model, tokenizer, output_processing="last_hidden", metric=metric)

def test_SemCSE_Euclidean():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("CLAUSE-Bielefeld/SemCSE")
    model = AutoModel.from_pretrained("CLAUSE-Bielefeld/SemCSE").to("cuda")

    from semantic_benchmark import evaluate_model_semantics
    evaluate_model_semantics(model, tokenizer, output_processing="last_hidden", metric="euclidean")

def test_SemCSE_cosine():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("CLAUSE-Bielefeld/SemCSE_cosine")
    model = AutoModel.from_pretrained("CLAUSE-Bielefeld/SemCSE_cosine").to("cuda")

    from semantic_benchmark import evaluate_model_semantics
    evaluate_model_semantics(model, tokenizer, output_processing="last_hidden", metric="cosine")


if __name__ == '__main__':
    #train_SemCSE()
    test_SemCSE_Euclidean()
    test_SemCSE_cosine()