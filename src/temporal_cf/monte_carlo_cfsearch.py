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

        mcts_solver = mcts(timeLimit=1000)
        best_action = mcts_solver.search(initialState=mcts_init)

        terminal = False
        curr = mcts_solver.root
        while not terminal:
            children = curr.children

            if len(children):
                best_child = mcts_solver.getBestChild(curr, 0)
            else:
                return {'cf': [curr.state._state],
                        'value': [curr.totalReward],
                        'terminal': [curr.state.isTerminal()]}

            curr = best_child
            terminal = curr.state.isTerminal()

            if terminal:
                cf = list(best_child.state._state)
                return {'cf': [cf],
                        'value': [best_child.totalReward],
                        'terminal': [terminal]}
