import torch
import torch.nn as nn

class LSTM_Autoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim=64):
        super(LSTM_Autoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(self.seq_len, 1, 1).permute(1,0,2)
        out, _ = self.decoder(h)
        return out
