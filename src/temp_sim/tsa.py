from pymoo.algorithms.moo.nsga2 import NSGA2

from src.temp_sim.ReplayBuffer import ReplayBuffer
import networkx as nx
import numpy as np
from pymoo.optimize import minimize

class TSA():



    def __init__(self, env_model, bb_model, target_action):
        self.env_model = env_model
        self.bb_model = bb_model
        self.target_action = target_action

        # parameters
        self.n_episodes = 100
        self.n_steps = 5

    def generate_counterfactuals(self, fact):
        # define problem
        problem = TSAProblem(fact)

        # define algorithm
        algorithm = NSGA2(pop_size=100)

        # optimize
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 200),
                       seed=1,
                       verbose=False)

        return res
