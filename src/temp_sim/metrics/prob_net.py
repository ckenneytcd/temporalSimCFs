import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ProbNet(nn.Module):

    def __init__(self):
        super(ProbNet, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=13, embedding_dim=50)
        self.fc1 = nn.Linear(50, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.output(x)

        return x

    def predict(self, x):
        output = self.forward(x)

        return np.round(output)
