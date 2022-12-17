import torch
import torch.nn as nn


class TextEmbeddingLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, tem_size):
        super(TextEmbeddingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.tem = nn.Linear(hidden_size * 2, tem_size)

    def forward(self, texts):
        embeddings = self.embed(texts)

        h0 = torch.zeros(self.num_layers * 2, texts.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, texts.size(0), self.hidden_size)

        outputs, (hidden, cell) = self.lstm(embeddings, (h0, c0))

        text_embedding = self.tem(hidden)

        return text_embedding
