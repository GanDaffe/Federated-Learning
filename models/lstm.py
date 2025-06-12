import torch 
from torch import nn 

class LSTM_header(nn.Module):

    def __init__(self):
        super(LSTM_header, self).__init__()
        self.embedding = nn.Embedding(2000, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
      
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class LSTM(nn.Module):
    def __init__(LSTM, self):
        super(LSTM, self).__init__()
        self.encode = LSTM_header() 
        self.classification = nn.Linear(256, 2) 
    
    def forward(self, x): 
        x = self.encode(x) 
        x = self.classification(x) 

        return x