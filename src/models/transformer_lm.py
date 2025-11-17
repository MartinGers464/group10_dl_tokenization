import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, model_dim, n_heads, n_layers, context_length):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(context_length, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.fc = nn.Linear(model_dim, vocab_size)

    def forward(self, x):
        b, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(b, 1)

        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.fc(h)
