import torch
import os
from transformers import AutoModel, AutoTokenizer
from settings import Config
from load_data import relative_path

os.makedirs(relative_path("data/saved_models"), exist_ok=True)

def load_model(load_weights_if_available=True):
    print(f"Loading model... ", end="")

    model = AutoModel.from_pretrained(Config.encoder_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(Config.encoder_checkpoint)
    model_file = f"{Config.save_name}_weights.pkl"

    if load_weights_if_available:
        try:
            state_dict = torch.load(os.path.join(os.path.dirname(__file__), f"data/saved_models/{model_file}"))
            model.load_state_dict(state_dict)
            print("Loaded existing model weights! ", end="")
        except Exception as e:
            print(e)
            print(f"No model weights found! ", end="")

    model = model.to('cuda')
    return model, tokenizer

def save_model(model):
    model_file = f"{Config.save_name}_weights.pkl"
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), f"data/saved_models/{model_file}"))