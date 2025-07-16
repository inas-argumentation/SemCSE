import torch

# Allows for writing x.np() to convert a cuda tensor directly to numpy
def to_np(self):
    return self.detach().cpu().numpy()
setattr(torch.Tensor, "np", to_np)

# Model settings
SENTENCE_BATCH_SIZE = 32
ABSTRACT_BATCH_SIZE = 8

class Config:
    save_name = None
    encoder_checkpoint = None

def set_save_name(save_name):
    Config.save_name = save_name

def set_encoder_checkpoint(encoder_checkpoint):
    Config.encoder_checkpoint = encoder_checkpoint

set_save_name("SemCSE")
set_encoder_checkpoint("KISTI-AI/Scideberta-full")

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

# Train the model either using triplet loss with Euclidean distance or with cosine similarity and softmax-based contrastive loss
metric = ["euclidean", "cosine"][0]