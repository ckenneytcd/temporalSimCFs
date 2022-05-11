from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection
import numpy as np
from src.temp_sim.tsa_problem import TSAProblem


class TSA:

    def __init__(self, env_model, bb_model, target_action, graph_path, buffer_path):
        self.env_model = env_model
        self.bb_model = bb_model
        self.target_action = target_action
        self.graph_path = graph_path
        self.buffer_path = buffer_path

        # params
        self.pop_size = 100
        self.num_generations = 20

    def get_counterfactuals(self, fact):
        # TODO: make sure each baseline also takes in one fact in the form of the dataframe row
        # Fact is passed as a df, need to extract the values
        fact = fact.values

        fact = fact.squeeze()

        # define problem
        problem = TSAProblem(fact, self.bb_model, self.target_action, self.env_model, self.graph_path, self.buffer_path)

        # extending fact to pop size
        fact_pop = problem.replay_buffer.state_buffer[0:self.pop_size]

        # define algorithm
        algorithm = NSGA2(pop_size=self.pop_size,
                          selection=get_selection('random'),
                          crossover=get_crossover("int_sbx"),
                          mutation=get_mutation("int_pm"),
                          eliminate_duplicates=True,
                          sampling=fact_pop)

        # optimize
        print('Optimizing...')
        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.num_generations),
                       seed=1,
                       verbose=True)

        print('Function value: {}'.format(res.F))
        print('Constraints violation: {}'.format(res.G))
        return res.X.tolist()
