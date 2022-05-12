import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 65*2
EMBEDDING_DIM = 10
BATCH_SIZE = 1

class ProbNet(nn.Module):

    def __init__(self):
        super(ProbNet, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=13, embedding_dim=EMBEDDING_DIM, )
        self.fc1 = nn.Linear(INPUT_SIZE * EMBEDDING_DIM, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x).view((BATCH_SIZE, -1))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.output(x)

        return x

    def predict(self, x):
        output = self.forward(x.unsqueeze(0))

        return torch.round(output.squeeze())
