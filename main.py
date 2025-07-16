from settings import set_save_name, set_encoder_checkpoint, metric

if __name__ == '__main__':
    set_save_name("SemCSE_run_test")
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
