import torch
import torch.nn as nn

class LSTM_Header(nn.Module):
    def __init__(self):
        super(LSTM_Header, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=2000, embedding_dim=50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, 50)
        _, (hn, _) = self.lstm(x)  # hn: (1, batch_size, 64)
        x = hn.squeeze(0)  # (batch_size, 64)
        x = self.fc1(x)    # (batch_size, 256)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.encode = LSTM_Header()
        self.classification = nn.Linear(256, 4)

    def forward(self, x):
        x = self.encode(x)
        x = self.classification(x)
        return x
