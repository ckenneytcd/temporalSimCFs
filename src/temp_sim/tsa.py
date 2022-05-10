from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from src.temp_sim.tsa_problem import TSAProblem


class TSA:

    def __init__(self, env_model, bb_model, target_action):
        self.env_model = env_model
        self.bb_model = bb_model
        self.target_action = target_action

    def get_counterfactuals(self, fact):
        # TODO: make sure each baseline also takes in one fact in the form of the dataframe row
        # Fact is passed as a df, need to extract the values
        fact = fact.values.squeeze()

        # define problem
        problem = TSAProblem(fact, self.bb_model, self.target_action, self.env_model,)

        # define algorithm
        algorithm = NSGA2(pop_size=100)

        # optimize
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                       seed=1,
                       verbose=False)

        return res
