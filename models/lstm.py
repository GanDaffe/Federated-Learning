import torch
import torch.nn as nn


class LSTM_Header(nn.Module):
    def __init__(self, vocab_size=2000):
        super(LSTM_Header, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=8)
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=256,
            num_layers=2,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x: (batch_size, seq_len) -> (batch_size, seq_len, 8)
        x = self.embedding(x)
        # output: (batch_size, seq_len, 256)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]  # shape: (batch_size, 256)
        x = self.dropout(x)
        return x

class LSTM(nn.Module):
    def __init__(self, vocab_size=2000, num_classes=4):
        super(LSTM, self).__init__()
        self.encode = LSTM_Header(vocab_size)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encode(x)  # (batch_size, 256)
        x = self.classifier(x)  # (batch_size, num_classes)
        return x

