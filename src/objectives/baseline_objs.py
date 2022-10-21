from math import sqrt

import torch


class BaselineObjectives:

    def __init__(self, env, bb_model, enc, enc_dataset, n_var):
        self.env = env
        self.bb_model = bb_model
        self.enc = enc
        self.n_var = n_var
        self.enc_dataset = enc_dataset

    def get_objectives(self, fact):
        return {
            'proximity': lambda x: self.proximity(x, fact),
            'sparsity': lambda x: self.sparsity(x, fact),
            'dmc': lambda x: self.data_manifold_closeness(x, self.enc_dataset)
        }

    def get_constraints(self, fact, target):
        return {
            'validity': lambda x: abs(self.bb_model.predict(x) - target),  # validity
            'realistic': lambda x: 1 - self.env.realistic(x),  # realistic
            'actionable': lambda x: 1 - self.env.actionable(x, fact)  # actionable
        }

    def proximity(self, x, fact):
            x_tensor = torch.tensor(x).squeeze()
            enc_x = self.enc.encode(x_tensor)

            fact_tensor = torch.tensor(fact).squeeze()
            enc_fact = self.enc.encode(fact_tensor)

            diff = abs(torch.subtract(enc_x, enc_fact))

            return sum(diff).item()

    def sparsity(self, x, fact):
        return (sum(fact[:25] != x[:25]) * 1.0).item()

    def data_manifold_closeness(self, x, data):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.enc.encode(x_tensor)
        recon = self.enc.decode(enc_x)

        diff = sqrt(sum((recon - x_tensor)**2))

        return diff