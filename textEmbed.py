import torch
import torch.nn as nn


class TextEmbeddingLSTM(nn.Module):
    def __init__(self, embedding_layer, embed_size, hidden_size, num_layers, tem_size):
        super(TextEmbeddingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = embedding_layer
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.tem = nn.Linear(hidden_size * 2, tem_size)

    def forward(self, desc_tokens):
        embeddings = self.embed(desc_tokens)

        h0 = torch.zeros(self.num_layers * 2, desc_tokens.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, desc_tokens.size(0), self.hidden_size)

        outputs, _ = self.lstm(embeddings, (h0, c0))

        h_sum = torch.sum(outputs, dim=1)
        text_embedding = self.tem(h_sum)
        return text_embedding


if __name__ == "__main__":
    # textEmbedding = TextEmbeddingLSTM(300, 128, 100000, 2, 300)
    # print(textEmbedding(torch.tensor([[1, 2, 3, 4, 5]])).shape)
    pass
