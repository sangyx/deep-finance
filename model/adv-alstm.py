import torch
from torch import nn

class TemporalAttetion(nn.Module):
    def __init__(self, input_size):
        super(TemporalAttetion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, 1, bias=False)

    def forward(self, x):
        alpha = self.linear2(torch.tanh(self.linear1(x)))
        alpha = torch.softmax(alpha, dim=1)
        a = torch.sum(alpha * x, dim=1)
        return a

class ADVALSTM(nn.Module):
    def __init__(self, input_size, map_size, hidden_size, epsilon, dropout=0.5):
        super(ADVALSTM, self).__init__()
        self.feature_mapping = nn.Linear(input_size, map_size)
        self.lstm = nn.LSTM(map_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(2 * hidden_size)
        self.attn = TemporalAttetion(hidden_size)
        self.linear = nn.Linear(2 * hidden_size, 1)
        self.epsilon = epsilon

    def gen_adv_samples(self, x, data_grad):
        adv_sample = x.data + self.epsilon * data_grad.data
        return adv_sample

    def forward(self, x):
        m = torch.tanh(self.feature_mapping(x))
        ht, (hn, _) = self.lstm(m)
        hn = hn.squeeze(dim=0)
        ht = self.dropout(ht)
        a = self.attn(ht)
        e = torch.cat([a, hn], dim=-1)
        e = self.norm(e)
        output = torch.sigmoid(self.linear(e))
        return output