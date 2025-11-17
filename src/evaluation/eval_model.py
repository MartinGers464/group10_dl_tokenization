import sys
import math
import torch
import torch.nn.functional as F

from src.config import config
from src.data.load_tinystories import load_tinystories
from src.data.make_dataloaders import make_dataloaders
from src.tokenization.factory import get_tokenizer
from src.models.transformer_lm import TransformerLM
from src.models.lstm_lm import LSTMLM


def build_model(tokenizer, model_type):
    vocab_size = tokenizer.vocab_size
    if model_type == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            model_dim=config["model_dim"],
            n_heads=config["num_heads"],
            n_layers=config["num_layers"],
            context_length=config["context_length"],
        )
    elif model_type == "lstm":
        model = LSTMLM(vocab_size, config["model_dim"])
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return model


def eval_val_loss(tokenizer_name, model_type, checkpoint_path):
    device = config["device"]
    tokenizer = get_tokenizer(tokenizer_name)

    # build model and load weights
    model = build_model(tokenizer, model_type)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # load data (same subset)
    train_ds, val_ds = load_tinystories()
    _, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_nll = total_loss / total_tokens          # nats per token
    ppl = math.exp(avg_nll)                      # perplexity
    print(f"Validation: nll/token={avg_nll:.4f}, perplexity={ppl:.4f}")


def main():
    if len(sys.argv) != 4:
        print("Usage: python -m src.evaluation.eval_model <tokenizer> <model_type> <checkpoint_path>")
        sys.exit(1)

    tok_name = sys.argv[1]       # e.g. "word"
    model_type = sys.argv[2]     # "transformer" or "lstm"
    ckpt = sys.argv[3]

    eval_val_loss(tok_name, model_type, ckpt)


if __name__ == "__main__":
    main()
