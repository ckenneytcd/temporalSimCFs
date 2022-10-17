import torch
from mcts import mcts

from src.baselines.encoder import VariationalAutoencoder
from src.objectives.baseline_objs import BaselineObjectives
from src.temporal_cf.mcts_state import MCTSState


class MCTSSearch:

    def __init__(self, env, bb_model, dataset):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim

        self.vae = VariationalAutoencoder(layers=[self.n_var, 128, 8])
        self.vae.fit(dataset)

        self.enc_data = self.vae.encode(torch.tensor(self.dataset.values))[0]

        self.obj = BaselineObjectives(env, bb_model, self.vae, self.enc_data, self.n_var)

    def generate_counterfactuals(self, fact, target):
        mcts_init = MCTSState(self.env, self.bb_model, target, fact, fact, self.obj)

        print('Running MCTS...')
        mcts_solver = mcts(iterationLimit=1000)
        mcts_solver.search(initialState=mcts_init)

        all_nodes = self.traverse(mcts_solver.root)
        potential_cf = [(n, n.totalReward) for n in all_nodes if n.isTerminal]

        return_dict = {
            'cf': [],
            'value': [],
            'terminal': []
        }

        for cf, cf_value in potential_cf:
            return_dict['cf'].append(cf)
            return_dict['value'].append(cf_value)
            return_dict['terminal'].append(cf.isTerminal)

        return return_dict

    def traverse(self, root):
        ''' Returns all nodes in the tree '''
        all_nodes = []

        if root is None:
            return None

        if root.children is None or len(root.children) == 0:
            return [root]

        if root.children is not None and len(root.children):
            for id, c in root.children.items():
                all_nodes = all_nodes + self.traverse(c)

        return all_nodes