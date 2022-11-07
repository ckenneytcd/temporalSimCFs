class CF:

    def __init__(self, cf_state, terminal, actions, cummulative_reward, value, searched_nodes, time):
        self.cf_state = cf_state
        self.terminal = terminal
        self.actions = actions
        self.cummulative_reward = cummulative_reward
        self.value = value
        self.searched_nodes = searched_nodes
        self.time = time
