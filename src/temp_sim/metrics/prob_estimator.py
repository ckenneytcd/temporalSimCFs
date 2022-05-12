import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

from src.temp_sim.metrics.prob_net import ProbNet


class ProbEstimator:

    def __init__(self, env, start_state, model_path):
        self.env = env
        self.start_state = start_state
        self.model_path = model_path

        self.n_steps = 10
        self.capacity = 10000
        self.train_size = int(0.8*self.capacity)
        self.test_size = int(0.2*self.capacity)

        self.setup()

    def setup(self):
        # TODO: save and load model
        try:
            self.net = self.load_model()
            print('Loaded model from file.')
        except FileNotFoundError:
            self.net = ProbNet()
            train_loader, test_loader = self.generate_data(self.env, self.start_state, self.capacity)
            self.train(train_loader, test_loader)

    def generate_data(self, env, start_state, capacity=10000):
        print('Generating distance data...')
        X = np.zeros((capacity, 65*2))
        y = np.zeros((capacity, 1))

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
        train_data, test_data = torch.utils.data.random_split(dataset, [self.train_size, self.test_size])

        print('Generate dataset with {} instances.'.format(len(dataset)))
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)  # create dataloader
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
        return train_loader, test_loader

    def train(self, train_loader, test_loader):
        print('Training probabilistic distance model...')
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.net.train()

        min_loss = 1000

        for epoch in range(20):  # loop over the dataset multiple times
            running_loss = 0.0
            for data in train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs).squeeze()
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5)
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            test_loss = self.evaluate(test_loader, epoch)
            print(f'[EPOCH {epoch + 1}] | Training loss: {running_loss / self.train_size:.3f} | Test loss: {test_loss:.3f}')
            if test_loss < min_loss:
                min_loss = test_loss
                self.save_model()

        print('Finished Training.')

    def evaluate(self, test_loader, epoch):
        self.net.eval()
        running_loss = 0.0
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            labels = labels.squeeze()

            # forward + backward + optimize
            outputs = self.net(inputs).squeeze()
            loss = mean_squared_error([outputs.detach().tolist()], [labels.detach().tolist()])

            running_loss += loss

        self.net.train()

        return running_loss / self.test_size

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
