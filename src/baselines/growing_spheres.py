import numpy as np

from src.envs.chess.chess_util import generate_neighborhood


class GrowingSpheres:
    '''
    Growing Spheres algorithm
    Implemented according to https://github.com/thibaultlaugel/growingspheres
    '''

    def __init__(self, bb_model, target_action=None):
        self.bb_model = bb_model
        self.n_search_samples = 1000
        self.p_norm = 2
        self.step = 0.002
        self.max_iter = 100
        self.target_action = target_action

    def get_counterfactuals(self, fact):
        # init step size for growing the sphere
        low = 0
        high = low + self.step

        # counter
        count = 0
        counter_step = 1

        # original label y
        y = self.bb_model.predict(fact.values)

        cf_found = False
        cf_star = None

        while not cf_found and count < self.max_iter:
            count = count + counter_step

            # Sample points on hyper sphere around instance
            candidate_cf = generate_neighborhood(fact.values, self.n_search_samples, self.step, count).values

            # Compute metrics to original instance
            dist = np.sum(candidate_cf == fact.values, axis=1) / fact.values.shape[1]

            # get y' for each counterfactual candidate
            cf_y = []
            for cf in candidate_cf:
                cf_y.append(self.bb_model.predict(cf))
                # print(cf_y)

            # filter cfs which satisfy prediction constraint
            if self.target_action is not None:
                indices = np.where(np.array(cf_y) == self.target_action)
            else:
                indices = np.where(np.array(cf_y) != y)

            candidate_cf = candidate_cf[indices]
            candidate_dist = dist[indices]

            if len(candidate_dist) > 0:  # certain candidates generated
                # select cf with min distance
                min_index = np.argmin(candidate_dist)
                cf_star = candidate_cf[min_index]
                cf_found = True

            # no candidate found & expand search radius
            low = high
            high = low + self.step

        if count % 10 == 0:
            print('Finished {} iterations.'.format(count))

        if cf_star is not None:
            return [cf_star]  # has to be returned as a list

        else:
            return []
