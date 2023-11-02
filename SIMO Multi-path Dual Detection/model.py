from rezero.transformer import RZTXEncoderLayer
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class Stransformers(nn.Module):
    def __init__(self, d_obj, nhead, num_layer = 6, d_embed = 128, d_class = 4):
        super(Stransformers, self).__init__()
        self.embedding = nn.Linear(d_obj, d_embed)
        self.denoise = nn.Linear(d_embed, 256)
        self.denoise2 = nn.Linear(256, d_embed)
        self.pos_encoder = PositionalEncoding(d_embed)

        enc_layer = RZTXEncoderLayer(d_embed, nhead)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layer)
        self.fc = nn.Linear(d_embed, d_class)

    def forward(self, x):
        y = self.embedding(x)
        y = nn.ReLU()(y)
        y = self.denoise(y)
        y = nn.ReLU()(y)
        y = self.denoise2(y)
        y = nn.ReLU()(y)
        y = self.pos_encoder(y)
        y = self.encoder(y)
        y = nn.ReLU()(y)
        y = self.fc(y[-1, :, :])
        out = torch.log_softmax(y, dim = -1)
        return out
    
def cross_entropy(y_true,y_pred):
    C=0
    # one-hot encoding
    for col in range(y_true.shape[-1]):
        y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
        y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
        C+=y_true[col]*torch.log(y_pred[col])+(1-y_true[col])*torch.log(1-y_pred[col])
    return -C