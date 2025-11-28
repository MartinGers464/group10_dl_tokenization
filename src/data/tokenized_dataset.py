import torch
from torch.utils.data import Dataset

class LMTokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, context_length, stride=16):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

        self.num_sequences = (len(self.ids) - context_length - 1) // self.stride

    def __len__(self):
        return len(self.ids) - self.context_length

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.context_length]
        y = self.ids[idx + 1 : idx + 1 + self.context_length]
        return x, y
