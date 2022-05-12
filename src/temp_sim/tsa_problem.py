import networkx as nx
import numpy as np
import autograd.numpy as anp
from pymoo.core.problem import Problem
import matplotlib.pyplot as plt

from src.temp_sim.metrics.graph_estimator import GraphEstimator
from src.temp_sim.metrics.prob_estimator import ProbEstimator

from src.temp_sim.metrics.replay_buffer import ReplayBuffer


class TSAProblem(Problem):

    def __init__(self, fact, bb_model, target_pred, env_model, graph_path, buffer_path, distance_mode='graph'):
        # TODO: not hardcoded parameters
        super().__init__(n_var=65, n_obj=3, n_constr=4, xl=0, xu=12, type_var=np.int64)
        self.fact = fact
        self.bb_model = bb_model
        self.target_pred = target_pred
        self.env_model = env_model
        self.graph_path = graph_path
        self.buffer_path = buffer_path
        self.distance_mode = distance_mode

        self.n_steps = 10
        self.fact_player = self.fact[-1]  # the last input is the player id (black/white)
        self.max_feature_changes = 5

        # set up the distance function predictor
        self.distance_estimator = self.setup_distance_model(self.distance_mode)

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
        # feature_change_constraint = sparsity > self.max_feature_changes
        # feature_change_constraint = shortest_path > self.max_feature_changes

        # concatenate the objectives
        out["F"] = anp.column_stack([shortest_path])
        out["G"] = anp.column_stack([validity, white_king_constraint, black_king_constraint, player_constraint])


    def setup_distance_model(self, distance_mode):
        print('Distance mode = {}'.format(distance_mode))
        if distance_mode == 'graph':
            distance_estimator = GraphEstimator(self.fact, self.env_model, self.buffer_path, self.graph_path)
            return distance_estimator

        elif distance_mode == 'prob':
            distance_estimator = ProbEstimator(self.env_model, self.fact)
            return distance_estimator


    def get_shortest_paths_for_cf_set(self, cfs):
        # TODO: optimize
        # Gets a shortest path from self.fact to a set of cfs
        n_cfs = cfs.shape[0]
        shortest_paths = np.zeros((n_cfs, ))

        for i in range(n_cfs):
            shortest_paths[i] = self.distance_estimator.get_shortest_path(self.fact, cfs[i])

        return shortest_paths



