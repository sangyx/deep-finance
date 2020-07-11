import torch
from torch import nn

class InputAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputAttention, self).__init__()
        self.linear1 = nn.Linear(input_size + 2 * hidden_size, input_size)
        self.linear2 = nn.Linear(input_size, 1)

    def forward(self, h, s, x):
        feature_dim = x.shape[-1]
        x = x.permute(2, 0, 1)
        h = h.unsqueeze(0).repeat(feature_dim, 1, 1)
        s = s.unsqueeze(0).repeat(feature_dim, 1, 1)
        u = self.linear1(torch.cat([x, h, s], dim=-1))
        e = self.linear2(torch.tanh(u))
        alpha = torch.softmax(e, dim=0).squeeze()
        return alpha.permute(1, 0)

class TemporalAttetion(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttetion, self).__init__()
        self.linear1 = nn.Linear(3 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, d, s, h):
        d = d.expand_as(h)
        s = s.expand_as(h)
        v = self.linear1(torch.cat([h, d, s], dim=-1))
        l = self.linear2(torch.tanh(v))
        beta = torch.softmax(l, dim=0)
        return beta

class DARNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, window_size, dropout, device):
        super(DARNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.window_size = window_size
        self.input_atten = InputAttention(window_size, hidden_size)
        self.encoder = nn.LSTMCell(input_size, hidden_size)
        self.decoder = nn.LSTMCell(1, hidden_size)
        self.temporal_atten = TemporalAttetion(hidden_size)
        self.linear1 = nn.Linear(1 + hidden_size, 1)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.h = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.es = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.d = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.ds = torch.zeros(batch_size, self.hidden_size, requires_grad=True).to(self.device)
        self.output = torch.zeros(batch_size, 1, requires_grad=True).to(device)

    def forward(self, X, Y=None):
        batch_size = X.shape[0]
        h, es, d, ds, output = self.h[: batch_size], self.es[: batch_size], self.d[: batch_size], self.ds[: batch_size], self.output[: batch_size]
        H = []
        for i in range(self.window_size):
            alpha = self.input_atten(h, es, X)
            xt = alpha * X[:, i, :]
            h, es = self.encoder(xt, (h, es))
            H.append(h)

        H = torch.stack(H).to(self.device)
        H = self.dropout(H)

        for i in range(self.window_size):
            beta = self.temporal_atten(d, ds, H)
            ct = torch.sum(beta * H, dim=0)
            if Y is None or i == 0:
                yt = self.linear1(torch.cat([output, ct], dim=-1))
            else:
                yt = self.linear1(torch.cat([Y[:, i-1], ct], dim=-1))
            d, ds = self.decoder(yt, (d, ds))
            output = self.linear2(torch.cat([ct, d], dim=-1))
            output = self.linear3(output)
            output = torch.sigmoid(output)
        return output