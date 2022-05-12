from carla.recourse_methods.catalog.face.library.face_method import build_constraints, build_graph, shortest_path, \
    choose_random_subset
import numpy as np
import pandas as pd

from src.envs.chess.chess_util import generate_neighborhood


class FACE:
    '''
     FACE algorithm
     Implemented according to https://github.com/sharmapulkit/FACE-Feasible-Actionable-Counterfactual-Explanations
    '''

    def __init__(self, data, bb_model, immutable_keys, target_action, mode='knn'):
        self.keys_immutable = immutable_keys
        self.data = data
        self.bb_model = bb_model
        self.mode = mode
        self.n_neighbors = 50
        self.radius = 0.25
        self.target_action = target_action
        self.frac = 0.4
        self.step = 0.002
        self.count = 1
        self.n_samples = 1000

    def get_counterfactuals(self, f):
        # for every factual instance
        for fact_ind in range(f.shape[0]):
            # search in the neighborhood of the instance
            neighborhood = generate_neighborhood(f.values[fact_ind], self.n_samples, self.step, self.count)

            # make sure factuals are at the beginning of the data
            cond = neighborhood.isin(f).values
            neighborhood = neighborhood.drop(neighborhood[cond].index)
            neighborhood = pd.concat(
                [f, neighborhood], ignore_index=True
            )

            # sample data subset for simplicity
            data = choose_random_subset(neighborhood, self.frac, fact_ind)

            # generate target predictions for instances in the neighbourhood
            y_hat = []
            for x in data.values:
                y_hat.append(self.bb_model.predict(x))

            # data instances with target action -- potential cfs
            y_positive_indices = np.where(np.array(y_hat) == self.target_action)

            # generating contraint matrix
            for i in range(len(self.keys_immutable)):
                immutable_constraint_matrix1, immutable_constraint_matrix2 = build_constraints(
                    data, i, self.keys_immutable
                )

            # setting up graphs
            if self.mode == "knn":
                boundary = 3  # chosen in ad-hoc fashion
                median = self.n_neighbors
                is_knn = True

            elif self.mode == "epsilon":
                boundary = 0.10  # chosen in ad-hoc fashion
                median = self.radius
                is_knn = False
            else:
                raise ValueError("Only possible values for mode are knn and epsilon")

            neighbors_list = [
                median - boundary,
                median,
                median + boundary,
            ]

            # build graph
            for n in neighbors_list:
                graph = build_graph(
                    data, immutable_constraint_matrix1, immutable_constraint_matrix2, is_knn, n
                )

            # generate shortest path from each instance to the fact
            distances, min_distance = shortest_path(graph, fact_ind)

            # candidate metrics from factual x
            candidate_min_distances = [
                min_distance,
                min_distance + 1
            ]

            min_distance_indices = np.array([0])
            for min_dist in candidate_min_distances:
                min_distance_indices = np.c_[
                    min_distance_indices, np.array(np.where(distances == min_dist))
                ]

            min_distance_indices = np.delete(min_distance_indices, 0)  # remove zeroes
            indeces_counterfactuals = np.intersect1d(
                np.array(y_positive_indices), np.array(min_distance_indices)  # cf are both predicting target class and min distance
            )

            # append all minimal distance cfs
            candidate_counterfactuals_star = []
            for i in range(indeces_counterfactuals.shape[0]):
                candidate_counterfactuals_star.append(data.values[indeces_counterfactuals[i]])

            if len(candidate_counterfactuals_star) > 0:
                print('Found counterfactual!')
            else:
                print('No counterfactuals found.')

            return candidate_counterfactuals_star