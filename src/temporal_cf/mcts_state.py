import copy
from copy import deepcopy

from src.objectives.baseline_objs import BaselineObjectives


class MCTSState:

    def __init__(self, env, model, target_action, fact, init_state, obj):
        self.env = env
        self.model = model
        self.target_action = target_action
        self.fact = fact
        self._state = init_state

        self.obj = obj

        self.lmbdas = {'proximity': 0.5,
                       'sparsity': 0.2,
                       'dmc': 0.3,
                       'validity': 10,
                       'realistic': 10,
                       'actionable': 10}

    def getPossibleActions(self):
        return self.env.get_actions(self._state)

    def takeAction(self, action):
        new_state = copy.copy(self)

        self.env.reset()
        self.env.set_state(self._state)
        obs, rew, done, _ = self.env.step(action)

        new_state._state = obs

        return new_state

    def isTerminal(self):
        best_action = self.model.predict(self._state)
        terminal = (best_action == self.target_action) or (self.env.check_done(self._state))

        return terminal

    def getReward(self):
        objectives = self.obj.get_objectives(self.fact)
        contraints = self.obj.get_constraints(self.fact, self.target_action)

        final_rew = 0.0

        for o_name, o_formula in objectives.items():
            final_rew += self.lmbdas[o_name] * o_formula(self._state)

        for c_name, c_formula in contraints.items():
            final_rew += self.lmbdas[c_name] * c_formula(self._state)

        return final_rew

