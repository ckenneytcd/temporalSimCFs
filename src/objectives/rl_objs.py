class BaselineObjectives:

    def __init__(self, env, bb_model, n_var, max_actions):
        self.env = env
        self.bb_model = bb_model
        self.n_var = n_var
        self.max_actions = max_actions

    def get_objectives(self, fact, actions, cummulative_rew):
        return {
            'cost': lambda x: self.cost(x, fact, actions, cummulative_rew),
            'reachability': lambda x: self.reachability(x, fact, actions),
            'stochasticity': lambda x: self.stochasticity(x, fact, actions)
        }

    def get_constraints(self, fact, target):
        return {
            'validity': lambda x: abs(self.bb_model.predict(x) - target),  # validity
            'realistic': lambda x: 1 - self.env.realistic(x),  # realistic
            'actionable': lambda x: 1 - self.env.actionable(x, fact)  # actionable
        }

    def cost(self, x, fact, actions, cummulative_rew):
        return cummulative_rew

    def reachability(self, x, fact, actions):
        return len(actions) * 1.0 / self.max_actions

    def stochasticity(self, x, fact, actions):
        # run simulations from fact with actions
        n_sim = 50

        cnt = 0
        for s in n_sim:
            self.env.reset()
            self.env.set_state(fact)

            done = False
            early_break = False
            for a in actions:
                if done:
                    early_break = True
                    break

                obs, rew, done, _ = self.env.step(a)

            if not early_break:
                # count how many times simulation ends up in x
                if self.env.equal_states(obs, x):
                    cnt += 1

        # percentage of ending up in x
        return (cnt*1.0) / n_sim
