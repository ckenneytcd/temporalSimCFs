from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_selection
import numpy as np

from src.envs.chess.chess_util import from_fen_to_board
from src.temp_sim.metrics.replay_buffer import ReplayBuffer
from src.temp_sim.tsa_problem import TSAProblem


class TSA:

    def __init__(self, env_model, bb_model, target_action, buffer_path):
        self.env_model = env_model
        self.bb_model = bb_model
        self.target_action = target_action
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
        problem = TSAProblem(fact,
                             self.bb_model,
                             self.target_action,
                             self.env_model,
                             self.buffer_path,
                             distance_mode='prob')

        # initializing the search
        replay_buffer = ReplayBuffer()
        replay_buffer.load(self.buffer_path)
        random_indices = np.random.choice(replay_buffer.state_buffer.shape[0], size=self.pop_size, replace=False)
        fact_pop = replay_buffer.state_buffer[random_indices, :]

        if np.any(np.all(from_fen_to_board('7k/2R5/6K1/8/8/7B/8/8 w - - 0 1') == fact_pop, axis=1)):
            print('Solution in initialization')

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
