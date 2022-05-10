import networkx as nx
import numpy as np
import autograd.numpy as anp
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt

from src.envs.chess.chess_util import from_board_to_fen
from src.temp_sim.ReplayBuffer import ReplayBuffer


class TSAProblem(Problem):

    def __init__(self, fact, bb_model, target_pred, env_model):
        # TODO: not hardcoded parameters
        super().__init__(n_var=65, n_obj=3, n_constr=1, xl=0, xu=12, type_var=np.int64)
        self.fact = fact
        self.bb_model = bb_model
        self.target_pred = target_pred
        self.env_model = env_model

        self.n_steps = 10
        self.fact_player = self.fact[-1] # the last input is the player id (black/white)

        # Fill replay buffer and generate graph
        self.replay_buffer = self.fill_replay_buffer(self.fact)
        self.G = self.generate_graph()
        # get all shortest paths between nodes
        self.paths = dict(nx.all_pairs_shortest_path(self.G))

    def _evaluate(self, x, out, *args, **kwargs):
        # check validity
        pred_cf = self.bb_model.get_output(x)  # comes as list
        target_list = [self.target_pred] * len(pred_cf)
        validity = 1 - np.array([x == y for x, y in zip(target_list, pred_cf)], dtype=int)

        # check sparsity (Hamming loss = number of feature changes)
        sparsity = np.sum(self.fact == x, axis=1)

        # check shortest path
        shortest_path = self.get_shortest_paths_for_cf_set(x)

        # constraint to not change player
        cf_player = x[:, -1]
        fact_player = np.full_like(cf_player, self.fact_player)
        player_constraint = np.abs(cf_player - fact_player)

        # concatenate the objectives
        out["F"] = anp.column_stack([validity, sparsity, shortest_path])
        out["G"] = anp.column_stack([player_constraint])

    def fill_replay_buffer(self, fact):
        # Generate dataset around f from env model
        replay_buffer = ReplayBuffer()
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
        return replay_buffer

    def generate_graph(self):
        # Generate graph from dataset
        node_list = self.replay_buffer.get_state_ids()
        edge_list = self.replay_buffer.get_edges()

        G = nx.Graph()

        # add nodes and edges
        G.add_nodes_from(node_list)
        G.add_edges_from(edge_list)

        # plot graph if small enough
        if len(G.nodes) < 100:
            nx.draw(G, with_labels=True, font_weight='bold')
            plt.show()
        print('Built graph with {} nodes and {} edges.'.format(len(G.nodes), len(G.edges)))
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



