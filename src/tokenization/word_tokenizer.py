import json
from collections import Counter
from pathlib import Path
import re

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

class WordTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(itos)
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @staticmethod
    def tokenize_text(text):
        # very simple: split on non-letters/numbers
        return [t for t in re.split(r"\s+", text.strip()) if t]

    @classmethod
    def train(cls, corpus_path, vocab_size=5000, save_dir="data/tokenizers/word"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        counter = Counter()
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                tokens = cls.tokenize_text(line)
                counter.update(tokens)

        # reserve space for specials
        max_words = vocab_size - len(SPECIAL_TOKENS)
        most_common = [w for w, _ in counter.most_common(max_words)]

        itos = SPECIAL_TOKENS + most_common
        stoi = {w: i for i, w in enumerate(itos)}

        # save vocab
        with open(save_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(itos, f, ensure_ascii=False, indent=2)

        return cls(stoi, itos)

    @classmethod
    def load(cls, save_dir="data/tokenizers/word"):
        save_dir = Path(save_dir)
        with open(save_dir / "vocab.json", encoding="utf-8") as f:
            itos = json.load(f)
        stoi = {w: i for i, w in enumerate(itos)}
        return cls(stoi, itos)

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize_text(text)
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        words = []
        for i in ids:
            if skip_special_tokens and self.itos[i] in SPECIAL_TOKENS:
                continue
            words.append(self.itos[i])
        return " ".join(words)
