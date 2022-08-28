import torch
from torch import nn


class SimpleLstm(nn.Module):
    def __init__(self, num_embeddings: int, hidden_size: int = 64, num_class: int = 2) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, dropout=.5, num_layers=4,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.Dropout(.5),
            nn.GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(32, num_class),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        embedding = self.emb(input_vector)
        hidden_state, _ = self.lstm(embedding)
        logit = self.fc(hidden_state)
        return self.output(logit[:, -1])
