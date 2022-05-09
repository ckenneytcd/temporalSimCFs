from src.temp_sim.ReplayBuffer import ReplayBuffer
import networkx as nx
import numpy as np


class TSA():

    def __init__(self, env_model, bb_model, target_action):
        self.env_model = env_model
        self.bb_model = bb_model
        self.target_action = target_action

        # parameters
        self.n_episodes = 100
        self.n_steps = 5

    def generate_counterfactuals(self, facts):
        self.replay_buffer = ReplayBuffer()

        # Define the loss function

        # Optimize the loss function
        pass

    def define_loss_function(self, f, cf):
        # Check validity

        # Check sparsity (Hamming loss)

        # Check shortest path

        pass

    def get_shortest_path(self, f, cf):
        # Generate dataset around f from env model
        print('Generating data around the factual instance')
        for i in range(1000):
            self.env_model.set_state(f)
            done = False
            step = 0
            curr_state = f
            while not done and (step <= self.n_steps) and not self.replay_buffer.is_full():
                rand_action = self.env_model.sample_action()
                next_state, rew, done, _ = self.env_model.step(rand_action)
                self.replay_buffer.add(curr_state, next_state)
                curr_state = next_state
                step += 1

        # Generate graph from dataset
        node_list = self.replay_buffer.get_state_ids()
        edge_list = self.replay_buffer.get_edges()

        G = nx.Graph()

        # add nodes and edges
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        # Get shortest path from f to cf
        f_id = self.replay_buffer.get_id_of_state(f)
        cf_id = self.replay_buffer.get_id_of_state(cf)

        # get all shortest paths between nodes
        s_paths = dict(nx.all_pairs_shortest_path(G))
        try:
            # if cf_id is in the list of shortest paths for f_id
            shortest_path = s_paths[f_id][cf_id]
        except KeyError():
            shortest_path = np.inf

        return shortest_path


    def optimize(self):
        pass