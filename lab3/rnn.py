from torch import relu
from torch.nn import Module, Linear, RNN, LSTM, GRU


class Rnn(Module):
    def __init__(self, embeddings, **kwargs):
        super().__init__()
        mode = kwargs.pop('mode', 'rnn')
        hidden_size = kwargs.pop('hidden_size', 300)
        bidirectional = kwargs.pop('bidirectional', False)
        layers = kwargs.pop('layers', 1)
        dropout = kwargs.pop('dropout', 1)

        self.embeddings = embeddings
        if mode == 'rnn':
            self.rnn = RNN(input_size=300, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=layers,
                           batch_first=False, dropout=dropout, bias=True)
        elif mode == 'lstm':
            self.rnn = LSTM(input_size=300, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=layers,
                            batch_first=False, dropout=dropout, bias=True)
        elif mode == 'gru':
            self.rnn = GRU(input_size=300, hidden_size=hidden_size, bidirectional=bidirectional, num_layers=layers,
                           batch_first=False, dropout=dropout, bias=True)
        in_features = hidden_size * (2 if bidirectional else 1)
        self.fc1 = Linear(in_features=in_features, out_features=150, bias=True)
        self.fc2 = Linear(in_features=150, out_features=1, bias=True)
        self.relu = relu

    def forward(self, data):
        x = self.embeddings(data)
        x = x.transpose(0, 1)
        x = x.float()
        x, _ = self.rnn(x, None)
        x = self.fc1(x[-1])
        x = self.relu(x)
        x = self.fc2(x)
        return x
