import networkx as nx
import numpy as np
import autograd.numpy as anp


from src.temp_sim.ReplayBuffer import ReplayBuffer


class TSAProblem():

    def __init__(self, fact, bb_model, target_pred, env_model):
        super().__init__(n_var=64, n_obj=2, n_constr=0, xl=0.0, xu=12.0)
        self.fact = fact
        self.bb_model = self.bb_model
        self.target_pred = target_pred
        self.env_model = env_model

        # Fill replay buffer and generate graph
        self.replay_buffer = self.fill_replay_buffer(self.fact)
        self.G = self.generate_graph()
        # get all shortest paths between nodes
        self.paths = dict(nx.all_pairs_shortest_path(self.G))

    def _evaluate(self, x, out, *args, **kwargs):
        # check validity
        pred_cf = self.bb_model.get_output(x)
        target_array = np.full_like(pred_cf, self.target_pred)
        validity = np.sum(pred_cf == target_array, axis=1)

        # check sparsity
        # TODO

        # check shortest path
        shortest_path = self.get_shortest_paths_for_cf_set(x)

        # concatenate the objectives
        out["F"] = anp.column_stack([validity, shortest_path])


    def fill_replay_buffer(self, fact):
        # Generate dataset around f from env model
        replay_buffer = ReplayBuffer()
        print('Generating data around the factual instance')
        for i in range(1000):
            self.env_model.set_state(fact)
            done = False
            step = 0
            curr_state = fact
            while not done and (step <= self.n_steps) and not self.replay_buffer.is_full():
                rand_action = self.env_model.sample_action()
                next_state, rew, done, _ = self.env_model.step(rand_action)
                replay_buffer.add(curr_state, next_state)
                curr_state = next_state
                step += 1

        return replay_buffer

    def generate_graph(self):
        # Generate graph from dataset
        node_list = self.replay_buffer.get_state_ids()
        edge_list = self.replay_buffer.get_edges()

        G = nx.Graph()

        # add nodes and edges
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        return G

    def get_shortest_path(self, f, cf):
        # Get shortest path from f to cf
        f_id = self.replay_buffer.get_id_of_state(f)
        cf_id = self.replay_buffer.get_id_of_state(cf)

        try:
            # if cf_id is in the list of shortest paths for f_id
            shortest_path = self.paths[f_id][cf_id]
        except KeyError():
            shortest_path = np.inf

        return shortest_path

    def get_shortest_paths_for_cf_set(self, cfs):
        # TODO: optimize
        # Gets a shortest path from self.fact to a set of cfs
        n_cfs = cfs.shape[0]
        shortest_paths = np.zeros((n_cfs, ))

        for i in range(n_cfs):
            shortest_paths[i] = self.get_shortest_path(self.fact, cfs[i])

        return shortest_paths



