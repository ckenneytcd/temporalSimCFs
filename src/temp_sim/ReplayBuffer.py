import numpy as np


class ReplayBuffer():

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.count = 0
        self.state_count = 0
        self.edge_buffer = np.zeros((capacity, 2))
        self.state_buffer = np.zeros((2*capacity, 65))  # TODO: remove hardcoding

    def add(self, state, next_state):
        # add states in the state buffer
        if np.sum(np.all(state == self.state_buffer, axis=1)) == 0:
            self.state_buffer[self.state_count] = state
            self.state_count += 1

        if (np.sum(np.all(next_state == self.state_buffer, axis=1)) == 0):
            self.state_buffer[self.state_count] = next_state
            self.state_count += 1

        source_id = self.get_id_of_state(state)
        dest_id = self.get_id_of_state(next_state)

        if not np.any(np.all(self.edge_buffer[:] == [source_id, dest_id], axis=1)):
            self.edge_buffer[self.count, :] = [source_id, dest_id]
            self.count += 1

    def get_size(self):
        return self.count

    def is_full(self):
        return self.count >= self.capacity

    def get_state_ids(self):
        return np.arange(0, self.state_count)

    def get_edges(self):
        return self.edge_buffer

    def get_id_of_state(self, state):
        try:
            id = np.where(np.all(state == self.state_buffer, axis=1))[0][0]
            return id
        except IndexError:
            return -1

