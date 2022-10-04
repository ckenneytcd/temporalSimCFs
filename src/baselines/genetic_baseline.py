import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from src.baselines.encoder import VariationalAutoencoder


class GeneticBaseline:

    def __init__(self, env, bb_model, dataset, proximity='mse'):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim
        self.proximity_type = proximity

        self.vae = VariationalAutoencoder(layers=[self.n_var, 128, 8])
        self.vae.fit(dataset)

        self.encoded_ds = self.vae.encode(torch.tensor(self.dataset.values))[0]

    def generate_counterfactuals(self, fact, target):
        print('Generating counterfactuals...')
        if self.proximity_type == 'mse':
            objs = [
                lambda x: np.mean(abs(x - fact)),  # proximity
                lambda x: ((sum(fact != x) * 1.0) / self.n_var)  # sparsity
            ]
        elif self.proximity_type == 'vae':
            objs = [
                lambda x: self.proximity(x, fact),  # proximity
                lambda x: self.sparsity(x, fact),  # sparsity
                lambda x: self.data_manifold_closeness(x, self.encoded_ds)  # data manifold closeness
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

        algorithm = NSGA2(pop_size=500,
                          sampling=X,
                          crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                          eliminate_duplicates=True)

        res = minimize(problem,
                       algorithm,
                       ('n_gen', 10),
                       seed=1,
                       verbose=True)

        solutions = res.pop.get('X')
        F = res.pop.get('F')
        G = res.pop.get('G')

        cfs = []
        for i, s in enumerate(solutions):
            f = F[i]
            g = G[i]
            if sum(g) == 0:
                cfs.append((s, f, g))

        return cfs

    def proximity(self, x, fact):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.vae.encode(x_tensor)[0]

        fact_tensor = torch.tensor(fact).squeeze()
        enc_fact = self.vae.encode(fact_tensor)[0]

        diff = abs(torch.subtract(enc_x, enc_fact))

        return sum(diff)

    def sparsity(self, x, fact):
        return ((sum(fact != x) * 1.0) / self.n_var)

    def data_manifold_closeness(self, x, data):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.vae.encode(x_tensor)[0]
        diffs = [sum(abs(d - enc_x)) for d in data]

        return min(diffs)

