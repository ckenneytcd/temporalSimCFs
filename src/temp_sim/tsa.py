from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation

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
        problem = TSAProblem(fact, self.bb_model, self.target_action, self.env_model)

        # define algorithm
        algorithm = NSGA2(pop_size=500,
                          sampling=get_sampling("int_random"),  #  TODO: allow for specification of data types
                          crossover=get_crossover("real_sbx"),
                          mutation=get_mutation("int_pm"),
                          eliminate_duplicates=True)

        # optimize
        print('Optimizing...')
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 100),
                       seed=1,
                       verbose=True)

        return res.X.tolist()
