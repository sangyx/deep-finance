import torch
from torch import nn

class AdaSFMCell(nn.Module):
    '''
    Cell Class for the Adaptive SFM RNN Layer
    '''
    def __init__(self, input_size, hidden_size, dropout):
        super(AdaSFMCell, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.lin_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_freq = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_state = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_g = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_omg = nn.Linear(input_size + hidden_size, hidden_size)
        self.lin_o = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.lin_z = nn.Linear(hidden_size, hidden_size)

    def outer(self, x, y):
        return x.unsqueeze(-1) * y.unsqueeze(1)

    def forward(self, x, t, z, ImS, ReS, omg):
        u = torch.cat([x, z], dim=-1)
        u = self.dropout(u)

        # The Joint State-Frequency Forget Gate
        freq = torch.sigmoid(self.lin_freq(u))
        state = torch.sigmoid(self.lin_state(u))
        fg = self.outer(freq, state)

        # Input Gates and Modulations
        i = torch.tanh(self.lin_i(u))
        g = torch.sigmoid(self.lin_g(u))

        # Updating State-Frequency Memory
        ReS = fg * ReS + self.outer(i * g, torch.cos(omg * t))
        ImS = fg * ImS + self.outer(i * g, torch.sin(omg * t))
        A = torch.sqrt(torch.pow(ReS, 2) + torch.pow(ImS, 2)) # (batch_size, frequency_components, state)

        z_n = torch.zeros_like(z)
        for k in range(A.shape[1]):
            o = torch.sigmoid(self.lin_o(torch.cat([A[:, k, :], x, z], dim=-1)))
            z_n += o * torch.tanh(self.lin_z(A[:, k, :]))

        z = z_n

        omega = self.lin_omg(u)

        return z, ImS, ReS, omega

class AdaSFM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, device):
        super(AdaSFM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = AdaSFMCell(input_size, hidden_size, dropout)
        self.device = device
        self.linear = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, X):
        batch_size = X.shape[0]
        z = torch.zeros(batch_size, self.hidden_size).to(self.device)
        omega = torch.zeros(batch_size, self.hidden_size).to(self.device)
        ImS = torch.zeros(batch_size, hidden_size, hidden_size).to(self.device)
        ReS = torch.zeros(batch_size, hidden_size, hidden_size).to(self.device)
        t = (torch.arange(self.hidden_size).float() + 1) / self.hidden_size

        for i in range(X.shape[1]):
            z, ImS, ReS, omega = self.cell(X[:, i, :], t[i], z, ImS, ReS, omega)
            z = self.norm(z)

        output = torch.sigmoid(self.linear(z))
        return output