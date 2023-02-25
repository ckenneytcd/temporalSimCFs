import pickle
import gym
import random
from collections import defaultdict

# The hyperparameters
alpha = 0.9
gamma = 0.6
epsilon = 0.1

NUM_EPISODES = 10000
# 100000

class TaxiBBModel():

    def __init__(self, env, model_path):
        self.model_path = model_path
        self.env = env
        self.model = self.load_model(model_path, env)

    def load_model(self, model_path, env):
        try:
            env = gym.make("Taxi-v3")
            print('Loaded bb model')
        except FileNotFoundError:
            print('Training bb model')
            q_table = defaultdict(int, {})
            q_table = self.train_agent(q_table, env, NUM_EPISODES)
            with open('q_table.pickle', "wb") as f:
                pickle.dump(dict(q_table), f)

        return env

    def predict(self, x):
        accq_path = 'q_table.pickle'
        with open(accq_path, 'rb') as f:
            q_table = pickle.load(f)
        action = self.select_optimal_action(q_table, x[0])
        return action
    

    def update(self,q_table, env, state):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = self.select_optimal_action(q_table, state)

        next_state, reward, _, _ = env.step(action)
        old_q_value = q_table[state][action]

        # Check if next_state has q values already
        if not q_table[next_state]:
            q_table[next_state] = {action: 0 for action in range(env.action_space.n)}

        # Maximum q_value for the actions in next state
        next_max = max(q_table[next_state].values())

        # Calculate the new q_value
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_max)

        # Finally, update the q_value
        q_table[state][action] = new_q_value

        return next_state, reward
    
    def train_agent(self, q_table, env, num_episodes):
        for i in range(num_episodes):
            state = env.reset()
            if not q_table[state]:
                q_table[state] = {
                    action: 0 for action in range(env.action_space.n)}

            epochs = 0
            num_penalties, reward, total_reward = 0, 0, 0
            while reward != 20:
                state, reward = self.update(q_table, env, state)
                total_reward += reward

                if reward == -10:
                    num_penalties += 1

                epochs += 1
            print("\nTraining episode {}".format(i + 1))
            print("Time steps: {}, Penalties: {}, Reward: {}".format(epochs,
                                                                    num_penalties,
                                                                    total_reward))

        print("Training finished.\n")

        return q_table
    def select_optimal_action(self, q_table, state):
        max_q_value_action = None
        max_q_value = -100000

        if q_table[state]:
            for action, action_q_value in q_table[state].items():
                if action_q_value >= max_q_value:
                    max_q_value = action_q_value
                    max_q_value_action = action

        return max_q_value_action
