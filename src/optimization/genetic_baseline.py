import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.problems.functional import FunctionalProblem

from src.optimization.autoenc import AutoEncoder
from src.models.counterfactual import CF
from src.objectives.baseline_objs import BaselineObjectives


class GeneticBaseline:

    def __init__(self, env, bb_model, dataset, obj):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim

        self.obj = obj


    def generate_counterfactuals(self, fact, target):
        print('Generating counterfactuals...')

        X = np.tile(fact, (1000, 1))

        objs = list(self.obj.get_objectives(fact).values())
        constr_ieq = list(self.obj.get_constraints(fact, target).values())

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

        cfs = {}
        valid_cfs = []
        cf_values = []
        for i, s in enumerate(solutions):
            f = F[i]
            g = G[i]

            if sum(g) == 0:
                cf = CF(s, True, [], 0, f, 10, 0)
                valid_cfs.append(cf)
                cf_values.append(f)


        best_index = np.argmax(np.array(cf_values))
        best_cf = valid_cfs[best_index]
        return best_cf



