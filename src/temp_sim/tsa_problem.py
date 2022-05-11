import networkx as nx
import numpy as np
import autograd.numpy as anp
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt
from src.temp_sim.replay_buffer import ReplayBuffer


class TSAProblem(Problem):

    def __init__(self, fact, bb_model, target_pred, env_model, graph_path, buffer_path):
        # TODO: not hardcoded parameters
        super().__init__(n_var=65, n_obj=3, n_constr=4, xl=0, xu=12, type_var=np.int64)
        self.fact = fact
        self.bb_model = bb_model
        self.target_pred = target_pred
        self.env_model = env_model
        self.graph_path = graph_path
        self.buffer_path = buffer_path

        self.n_steps = 10
        self.fact_player = self.fact[-1]  # the last input is the player id (black/white)
        self.max_feature_changes = 5

        # Fill replay buffer and generate graph
        self.replay_buffer = self.fill_replay_buffer(self.fact)
        self.G = self.generate_graph()
        # get all shortest paths from between nodes
        self.paths = dict(nx.all_pairs_shortest_path(self.G))


    def _evaluate(self, x, out, *args, **kwargs):
        # check validity
        pred_cf = self.bb_model.get_output(x)  # comes as list
        target_list = [self.target_pred] * len(pred_cf)
        validity = np.array([x != y for x, y in zip(target_list, pred_cf)], dtype=int)

        # check sparsity (Hamming loss = number of feature changes)
        sparsity = self.n_var - np.sum(self.fact == x, axis=1)

        # check shortest path
        shortest_path = self.get_shortest_paths_for_cf_set(x)

        # constraint to not change player
        cf_player = x[:, -1]
        fact_player = np.full_like(cf_player, self.fact_player)
        player_constraint = np.abs(cf_player - fact_player)

        # kings constraint
        white_king_constraint = 1 != np.count_nonzero(x == 6, axis=1)
        black_king_constraint = 1 != np.count_nonzero(x == 12, axis=1)

        # feature changes constrain
        feature_change_constraint = sparsity - self.max_feature_changes

        # concatenate the objectives
        out["F"] = anp.column_stack([validity, sparsity, shortest_path])
        out["G"] = anp.column_stack([white_king_constraint, black_king_constraint, feature_change_constraint, player_constraint])

    def fill_replay_buffer(self, fact):
        # Generate dataset around f from env model
        replay_buffer = ReplayBuffer()
        try:
            replay_buffer.load(self.buffer_path)
            print('Loaded replay buffer from: {}'.format(self.buffer_path))
        except FileNotFoundError:
            print('Generating data around the factual instance')
            for i in range(1000):
                self.env_model.set_state(fact)
                done = False
                step = 0
                curr_state = fact
                while not done and (step <= self.n_steps) and not replay_buffer.is_full():
                    rand_action = self.env_model.sample_action()
                    next_state, rew, done, _ = self.env_model.step(rand_action)
                    replay_buffer.add(curr_state, next_state)
                    curr_state = next_state
                    step += 1
            print('Finished. Generated {} instances.'.format(replay_buffer.count))
            replay_buffer.save(self.buffer_path)
            print('Saved buffer at: {}'.format(self.buffer_path))

        return replay_buffer

    def generate_graph(self):
        # Generate graph from dataset
        node_list = self.replay_buffer.get_state_ids()
        edge_list = self.replay_buffer.get_edges()

        G = nx.Graph()

        try:
            # load graph if it exists
            G = nx.read_gpickle(self.graph_path)
            print('Loaded graph from: {}'.format(self.graph_path))
        except FileNotFoundError:
            # add nodes and edges
            G.add_nodes_from(node_list)
            G.add_edges_from(edge_list)

            # plot graph if small enough
            if len(G.nodes) < 100:
                nx.draw(G, with_labels=True, font_weight='bold')
                plt.show()
            print('Built graph with {} nodes and {} edges.'.format(len(G.nodes), len(G.edges)))

            nx.write_gpickle(G, self.graph_path)
            print('Saved graph at: {}'.format(self.graph_path))

        return G

    def get_shortest_path(self, f, cf):
        # Get shortest path from f to cf
        f_id = self.replay_buffer.get_id_of_state(f)
        cf_id = self.replay_buffer.get_id_of_state(cf)

        if f_id == -1 or cf_id == -1:
            # if states don't exist in the graph - return large distance
            return +100

        try:
            # if cf_id is in the list of shortest paths for f_id
            shortest_path = self.paths[f_id][cf_id][0]
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



