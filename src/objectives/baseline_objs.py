import torch


class BaselineObjectives:

    def __init__(self, env, bb_model, vae, enc_dataset, n_var):
        self.env = env
        self.bb_model = bb_model
        self.vae = vae
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
        enc_x = self.vae.encode(x_tensor)[0]

        fact_tensor = torch.tensor(fact).squeeze()
        enc_fact = self.vae.encode(fact_tensor)[0]

        diff = abs(torch.subtract(enc_x, enc_fact))

        return sum(diff).item()

    def sparsity(self, x, fact):
        return ((sum(fact != x) * 1.0) / self.n_var).item()

    def data_manifold_closeness(self, x, data):
        x_tensor = torch.tensor(x).squeeze()
        enc_x = self.vae.encode(x_tensor)[0]
        diffs = [sum(abs(d - enc_x)) for d in data]

        return min(diffs).item()