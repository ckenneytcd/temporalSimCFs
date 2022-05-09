import numpy as np


class ReplayBuffer():

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.count = 0
        self.state_count = 0
        self.edge_buffer = np.array((capacity, 2))
        self.state_buffer = np.array((2*capacity, 65))  # TODO: remove hardcoding

    def add(self, state, next_state):
        # add states in the state buffer
        if not (state in list(self.state_buffer)):
            self.state_buffer[self.state_count] = state
            self.state_count += 1

        if not (next_state in list(self.state_buffer)):
            self.state_buffer[self.state_count] = next_state
            self.state_count += 1

        source_id = np.where(self.state_buffer == state)
        dest_id = np.where(self.state_buffer == next_state)

        if not any((self.edge_buffer[:] == [state, next_state]).all(1)):
            self.edge_buffer[self.count, :] = [source_id, dest_id]
            self.count += 1

    def get_size(self):
        return self.count

    def is_full(self):
        return self.count >= self.capacity


