import torch
from mcts import mcts
import numpy as np

from src.baselines.autoenc import AutoEncoder
from src.objectives.baseline_objs import BaselineObjectives
from src.temporal_cf.mcts_state import MCTSState


class MCTSSearch:

    def __init__(self, env, bb_model, dataset):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim

        self.vae = AutoEncoder(layers=[self.n_var, 128, 8])
        train_dataset = dataset.sample(frac=0.8, random_state=1)
        test_dataset = dataset.drop(train_dataset.index)
        self.vae.fit(train_dataset, test_dataset)

        self.enc_data = self.vae.encode(torch.tensor(self.dataset.values))[0]

        self.obj = BaselineObjectives(env, bb_model, self.vae, self.enc_data, self.n_var)

    def generate_counterfactuals(self, fact, target):
        mcts_init = MCTSState(self.env, self.bb_model, target, fact, fact, self.obj)

        print('Running MCTS...')
        mcts_solver = mcts(iterationLimit=50000, maxLevel=10)
        mcts_solver.search(initialState=mcts_init)

        all_nodes = self.traverse(mcts_solver.root)
        print('Expanded {} nodes'.format(len(all_nodes)))

        potential_cf = [(n.state, n.totalReward) for n in all_nodes if n.state.isTerminal()]

        return_dict = {
            'cf': [],
            'objectives': [],
            'value': [],
            'terminal': []
        }

        for cf, cf_value in potential_cf:
            if list(cf._state) not in return_dict['cf']:
                return_dict['cf'].append(list(cf._state))
                return_dict['objectives'].append(cf.getIndRews())
                return_dict['value'].append(cf.getReward())
                return_dict['terminal'].append(True)

        # return only the best one
        best_cf_ind = np.argmax(np.array(list(return_dict['value'])))
        best_cf = {
            'cf': [return_dict['cf'][best_cf_ind]],
            'objectives': [return_dict['objectives'][best_cf_ind]],
            'value': [return_dict['value'][best_cf_ind]],
            'terminal': [return_dict['terminal'][best_cf_ind]]
        }

        return best_cf

    def traverse(self, root, nodes=None):
        ''' Returns all nodes in the tree '''
        if nodes is None:
            nodes = set()

        nodes.add(root)

        if root.children is not None and len(root.children):
            for id, c in root.children.items():
                self.traverse(c, nodes)

        return nodes