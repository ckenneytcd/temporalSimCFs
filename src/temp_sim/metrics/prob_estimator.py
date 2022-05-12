import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from src.temp_sim.metrics.prob_net import ProbNet


class ProbEstimator:

    def __init__(self, env, start_state, model_path):
        self.env = env
        self.start_state = start_state
        self.model_path = model_path

        self.n_steps = 10
        self.capacity = 10000

        self.setup()

    def setup(self):
        # TODO: save and load model
        try:
            self.net = self.load_model()
            print('Loaded model from file.')
        except FileNotFoundError:
            self.net = ProbNet()
            dataloader = self.generate_data(self.env, self.start_state, self.capacity)
            self.train(dataloader)

    def generate_data(self, env, start_state, capacity=10000):
        print('Generating distance data...')
        X = np.zeros((capacity, 65*2))
        y = np.zeros((capacity, ))

        count = 0
        while count < capacity:
            env.set_state(start_state)
            dist = 0.0
            done = False
            curr_step = 0
            state = start_state
            while not done and curr_step < self.n_steps:
                rand_action = env.sample_action()
                new_state, rew, done, _ = env.step(rand_action)

                if count >= capacity:
                    break

                x_input = np.concatenate((state, new_state), axis=0)
                if not np.any(np.all(X == x_input, axis=1)):
                    X[count, :] = np.concatenate((state, new_state), axis=0)
                    y[count] = dist
                    count += 1

                curr_step += 1
                dist += 1
                state = new_state

        tensor_X = torch.Tensor(X)  # transform to torch tensor
        tensor_X = tensor_X.long()
        tensor_y = torch.Tensor(y)

        dataset = TensorDataset(tensor_X, tensor_y)  # create dataset
        print('Generate dataset with {} instances.'.format(len(dataset)))
        dataloader = DataLoader(dataset, batch_size=1)  # create dataloader

        return dataloader

    def train(self, dataloader):
        print('Training probabilistic distance model...')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.net.train()

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for data in dataloader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            print(f'[EPOCH {epoch + 1}] loss: {running_loss / 10000:.3f}')

        self.save_model()
        print('Finished Training.')
0
    def get_shortest_path(self, f, cf):
        state = np.concatenate((f, cf), 0)

        state = np.expand_dims(state, 0)

        state = torch.LongTensor(state).squeeze()

        shortest_path = self.net.predict(state)

        return shortest_path

    def save_model(self):
        torch.save(self.net.state_dict(), self.model_path)
        print('Saved model.')

    def load_model(self):
        model = ProbNet()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        return model
