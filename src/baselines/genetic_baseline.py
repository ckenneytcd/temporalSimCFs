import numpy as np
import torch
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core import problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from src.baselines.encoder import VariationalAutoencoder


class GeneticBaseline:

    def __init__(self, env, bb_model, dataset, proximity='mse' ):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim
        self.proximity = proximity

        self.vae = VariationalAutoencoder(layers=[self.n_var, 16])
        self.vae.fit(dataset)

    def generate_counterfactuals(self, fact, target):
        print('Generating counterfactuals...')
        if self.proximity == 'mse':
            objs = [
                lambda x: np.mean(abs(x - fact))
                          + ((sum(fact != x) * 1.0) / self.n_var)  # proximity  and sparsity
            ]
        elif self.proximity == 'vae':
            objs = [
                lambda x: torch.mean(torch.subtract(self.vae.encode(torch.tensor(x))[0], self.vae.encode(torch.tensor(fact))[0]**2)),  # proximity
                lambda x: ((sum(fact != x) * 1.0) / self.n_var),  # sparsity
                lambda x: min([abs(self.vae.encode(torch.tensor(d))[0] - self.vae.encode(torch.tensor(x))[0]) for d in self.dataset])   # data manifold closeness
            ]

        X = np.tile(fact, (1000, 1))

        constr_ieq = [
            lambda x: abs(self.bb_model.predict(x) - target),  # validity
            lambda x: 1 - self.env.realistic(x),  # realistic
            lambda x: 1 - self.env.actionable(x, fact)  # actionable
        ]

        problem = FunctionalProblem(self.n_var,
                                    objs,
                                    constr_ieq=constr_ieq,
                                    xl=self.env.lows,
                                    xu=self.env.highs)

        algorithm = GA(pop_size=100,
                       sampling=X,
                       crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                       mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                       eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 500),
                       seed=1,
                       verbose=False)

        solutions = res.pop.get('X')

        return solutions[0]