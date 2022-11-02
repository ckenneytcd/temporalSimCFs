import copy


class MCTSState:

    def __init__(self, env, model, target_action, fact, init_state, obj):
        self.env = env
        self.model = model
        self.target_action = target_action
        self.fact = fact
        self._state = init_state

        self.obj = obj

        self.lmbdas = {'proximity': -0.33,
                       'sparsity': -0.33,
                       'dmc': -0.33,
                       'validity': 0,
                       'realistic': -1,
                       'actionable': -1}  # because MCTS maximizes value

    def getPossibleActions(self):
        return self.env.get_actions(self._state)

    def takeAction(self, action):
        nns = []
        for i in range(2**5):
            new_state = copy.copy(self)

            self.env.reset()
            self.env.set_state(self._state)
            obs, rew, done, _ = self.env.step(action)

            new_state._state = copy.copy(obs)

            found = False
            for nn in nns:
                if self.env.equal_states(obs, nn._state):
                    found = True
                    break

            if not found:
                nns.append(new_state)

        return nns

    def isTerminal(self):
        best_action = self.model.predict(self._state)
        terminal = (best_action == self.target_action) or (self.env.check_done(self._state))

        return terminal

    def getReward(self):
        objectives = self.obj.get_objectives(self.fact)
        contraints = self.obj.get_constraints(self.fact, self.target_action)

        obj_rew = 0.0
        contr_rew = 0.0

        for o_name, o_formula in objectives.items():
            obj_rew += self.lmbdas[o_name] * o_formula(self._state)

        for c_name, c_formula in contraints.items():
            contr_rew += self.lmbdas[c_name] * c_formula(self._state)

        final_rew = contr_rew + obj_rew

        return final_rew

    def getIndRews(self):
        objectives = self.obj.get_objectives(self.fact)
        contraints = self.obj.get_constraints(self.fact, self.target_action)

        rewards = {}

        for o_name, o_formula in objectives.items():
            rewards[o_name] = self.lmbdas[o_name] * o_formula(self._state)

        for c_name, c_formula in contraints.items():
            rewards[c_name] = self.lmbdas[c_name] * c_formula(self._state)

        return rewards

    def get_cf_reward(self, state):
        objectives = self.obj.get_objectives(self.fact)
        contraints = self.obj.get_constraints(self.fact, self.target_action)

        final_rew = 0.0

        for o_name, o_formula in objectives.items():
            final_rew += self.lmbdas[o_name] * o_formula(state)

        for c_name, c_formula in contraints.items():
            final_rew += self.lmbdas[c_name] * c_formula(state)

        return final_rew

