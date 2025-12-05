import json
import sys
import math
import torch
import torch.nn.functional as F

from src.config import config
from src.data.load_tinystories import load_tinystories
from src.data.load_wiki2 import load_wiki2
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
            dropout=config["dropout"],
        )
    elif model_type == "lstm":
        model = LSTMLM(vocab_size, config["model_dim"])
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    return model


def eval_val_loss(tokenizer_name, model_type, checkpoint_path, split): # TODO! Change to test when needed
    device = config["device"]
    tokenizer = get_tokenizer(tokenizer_name)


    # build model and load weights
    model = build_model(tokenizer, model_type)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    pad_id = tokenizer.pad_id
    vocab_size = tokenizer.vocab_size

    # load data (same subset)
    train_ds, val_ds, test_ds = load_wiki2()
    #_, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)
    #train_loader, val_loader = make_dataloaders(train_ds, val_ds, tokenizer, config)

    if split == "val":
        eval_ds = val_ds
    elif split == "test":
        eval_ds = test_ds
    elif split == "train": #just a test that there is no issue with training set eval
        eval_ds = train_ds
    else:
        raise ValueError(f"Unknown split: {split} (expected 'val' or 'test')")

    train_raw, val_raw, test_raw = load_wiki2()
    # print("Loaded Wiki2:", len(train_raw), "train,", len(val_raw), "val", len(test_raw), "test")

    train_texts = [ex["text"] for ex in train_raw]
    val_texts   = [ex["text"] for ex in val_raw]
    test_texts  = [ex["text"] for ex in test_raw]

    print("Building tokenized datasets and dataloaders...")
    # NOTE: make_dataloaders must now also build test_loader, test_ds
    # _, val_loader = make_dataloaders(train_ds, eval_ds, tokenizer, config)

    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = make_dataloaders(train_raw, val_raw, test_raw, tokenizer, config)
    
    # count total characters in eval_ds
    total_chars = 0
    total_bytes = 0
    for ex in eval_ds:
        text = ex["text"]
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

    total_loss = 0.0
    total_tokens = 0

 # compute test metrics
    print("Evaluating on TEST set with final model...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tloss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=pad_id
            )
            test_loss += tloss.item()

    test_loss /= len(test_loader)
      # compute test metrics
    nll_token = test_loss
    ppl = math.exp(nll_token)
    bits_per_token = nll_token / math.log(2)

    total_chars_test = sum(len(t) for t in test_texts)
    total_tokens_test = len(test_ds.ids)
    tokens_per_char = total_tokens_test / total_chars_test

    nll_per_char = (nll_token * total_tokens_test) / total_chars_test
    bpc = nll_per_char / math.log(2)

    total_bytes_test = sum(len(t.encode("utf-8")) for t in test_texts)
    total_nll_test = nll_token * total_tokens_test        
    nll_per_byte = total_nll_test / total_bytes_test
    bits_per_byte = nll_per_byte / math.log(2.0)

    print("TEST metrics:")
    print(f"  [TEST] nll/token       = {nll_token:.4f} nats")
    print(f"  [TEST] perplexity      = {ppl:.4f}")
    print(f"  [TEST] bits per token  = {bits_per_token:.4f}")
    print(f"  [TEST] nll/char        = {nll_per_char:.4f} nats")
    print(f"  [TEST] bits per char   = {bpc:.4f} bits")
    print(f"  [TEST] tokens per char = {tokens_per_char:.4f}")
    print(f"  [TEST] bits per byte   = {bits_per_byte:.4f}")

    test_results = {
        "split": "test",
        "nll_token": nll_token,
        "perplexity": ppl,
        "bits_per_token": bits_per_token,
        "nll_per_char": nll_per_char,
        "bits_per_char": bpc,
        "tokens_per_char": tokens_per_char,
        "bits_per_byte": bits_per_byte,
    }
    with open(f"results/test_metrics_{tokenizer_name}_{model_type}.json", "w") as f:
        json.dump(test_results, f, indent=2)

def main():
    if len(sys.argv) != 5:
        print("Usage: python -m src.evaluation.eval_model <tokenizer> <model_type> <checkpoint_path> <split>")
        sys.exit(1)

    tok_name = sys.argv[1]       # e.g. "word"
    model_type = sys.argv[2]     # "transformer" or "lstm"
    ckpt = sys.argv[3]
    split = sys.argv[4]          # "val", "test", or "train"

    eval_val_loss(tok_name, model_type, ckpt, split)

    


if __name__ == "__main__":
    main()
