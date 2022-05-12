import copy
import numpy as np
import gym

from src.envs.chess.chess_util import from_board_to_fen, is_valid_fen

NUM_FEATURES = 64
PLAYER_POS = 64


class RelaceEnv(gym.Env):

    def __init__(self, target_action, fact, lmbda):
        self.state = np.zeros((NUM_FEATURES * 2 + 1, ))
        self.target_action = target_action
        self.fact = fact.values.squeeze()
        self.lmbda = lmbda

        self.prev_dist = 0
        self.max_change = 10

        lows = np.zeros((2*NUM_FEATURES + 1, ))
        highs = np.ones((2*NUM_FEATURES + 1, ))
        highs[0: NUM_FEATURES] = 11  # can't add kings
        # state consists of board features (64) + player feature (1) + change features (64)
        self.observation_space = gym.spaces.Box(low=lows, high=highs, shape=(2*NUM_FEATURES + 1, ))
        # action is a tuple -- which feature to change and to which value
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(NUM_FEATURES), gym.spaces.Discrete(13)))

    def set_bb_model(self, bb_model):
        self.bb_model = bb_model

    def reset(self):
        random_start_state = self.fact  # ignore the last feature about whose move it is
        self.set_x_state(random_start_state)

        self.prev_dist = 0
        return self.state

    def step(self, action):
        # action consists of feature and amount it should be changed
        feature = action['action_type']
        value = action['action_args']

        # value = affine_transform(value, min_val=0, max_val=11)
        #
        # value = round(value)

        # mark that feature has been changed
        self.state[feature + 65] = 1

        # update the feature value
        self.state[feature] = value

        reward, done = self.calculate_reward()

        return self.state, reward, done, {}

    def calculate_reward(self):
        # if action leads to an impossible state
        if not self.is_valid_state():
            return -100, True

        new_pred = self.bb_model.predict(self.state[0:(NUM_FEATURES + 1)])
        print(new_pred)

        # check if counterfactual is reached
        if new_pred == self.target_action:
            done = True
            reward = -self.lmbda * (self.hamming_loss(self.fact, self.state[0:NUM_FEATURES]) - self.prev_dist)
            self.prev_dist = self.hamming_loss(self.fact, self.state[0:NUM_FEATURES])
            return reward, done
        else:
            # calculate the reward
            reward = 1 - self.lmbda * (self.hamming_loss(self.fact, self.state[0:NUM_FEATURES]) - self.prev_dist)
            self.prev_dist = self.hamming_loss(self.fact, self.state[0:64])

            done = False
            # check if change limit is reached
            num_changed = np.sum(self.state[64:])
            if num_changed > self.max_change:
                done = True

            return reward, done

    def hamming_loss(self, a, b):
        return np.sum(a == b, axis=-1) / NUM_FEATURES

    def is_valid_state(self):
        board = self.state[0: 64]
        fen = from_board_to_fen(board)
        return is_valid_fen(fen)

    def close(self):
        pass

    def render(self):
        pass

    def set_x_state(self, x_state):
        self.state = np.zeros((NUM_FEATURES * 2 + 1, ))
        self.state[0: (NUM_FEATURES + 1)] = copy.copy(x_state)

