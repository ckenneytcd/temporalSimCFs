


class ReLACE():
    ''' ReLACE algorithm for generating counterfactual explanations using reinforcement learning '''

    def __init__(self, bb_model, data, target_action, immutable_keys):
        self.bb_model = bb_model
        self.data = data
        self.target_action = target_action
        self.immutable_keys = immutable_keys


    def train_RL_model(self):
        pass

    def get_counterfactuals(self, facts):
        pass