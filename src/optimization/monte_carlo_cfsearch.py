import numpy as np
from src.models.counterfactual import CF
from src.optimization.mcts import MCTS


class MCTSSearch:

    def __init__(self, env, bb_model, dataset, obj):
        self.env = env
        self.bb_model = bb_model
        self.dataset = dataset
        self.n_var = env.state_dim
        self.obj = obj

    def generate_counterfactuals(self, fact, target):
        mcts_solver = MCTS(self.env, self.bb_model, self.obj, fact, target, max_level=10)
        tree_size, time = mcts_solver.search(init_state=fact, num_iter=100)

        all_nodes = self.traverse(mcts_solver.root)
        print('Expanded {} nodes'.format(len(all_nodes)))

        potential_cf = [CF(n.state, True, n.prev_actions, n.cummulative_reward, n.get_reward(), tree_size, time)
                        for n in all_nodes if n.is_terminal()]

        print('Found {} valid counterfactuals'.format(len(potential_cf)))
        for cf in potential_cf:
            self.env.render_state(cf.cf_state)
            print('Value: {}'.format(cf.value))

        # return only the best one
        if len(potential_cf):
            best_cf_ind = np.argmax(np.array([cf.value for cf in potential_cf]))
            best_cf = potential_cf[best_cf_ind]
        else:
            return None

        return best_cf

    def traverse(self, root, nodes=None):
        ''' Returns all nodes in the tree '''
        if nodes is None:
            nodes = set()

        nodes.add(root)

        if root.children is not None and len(root.children):
            children = []
            for action in root.children.keys():
                children += root.children[action]

            for c in children:
                self.traverse(c, nodes)

        return nodes