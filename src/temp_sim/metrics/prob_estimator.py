import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from src.temp_sim.metrics.prob_net import ProbNet


class ProbEstimator:

    def __init__(self, env, start_state):
        self.env = env
        self.start_state = start_state

        self.n_steps = 10
        self.capacity = 10000

        self.net = ProbNet()

    def setup(self):
        dataloader = self.generate_data(self.env, self.start_state, self.capacity)
        self.train(dataloader)

    def generate_data(self, env, start_state, capacity=10000):
        print('Generating distance data')
        X = np.array((capacity, 65*2))
        y = np.array((capacity))

        count = 0
        while count < capacity:
            env.set_state(start_state)
            dist = 0
            done = False
            curr_step = 0
            state = start_state
            while not done and curr_step < self.n_steps:
                rand_action = env.sample_action(state)
                new_state, rew, done, _ = env.step(rand_action)

                X[count, :] = [state, new_state]
                y[count] = dist

                dist += 1
                count += 1
                curr_step += 1
                state = new_state

        tensor_X = torch.Tensor(X)  # transform to torch tensor
        tensor_y = torch.Tensor(y)

        dataset = TensorDataset(tensor_X, tensor_y)  # create dataset
        dataloader = DataLoader(dataset)  # create dataloader

        return dataloader

    def train(self, dataloader):
        print('Training probabilistic distance model')
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

