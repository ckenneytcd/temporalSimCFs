import copy
import random

import chess
import gym
import numpy as np
from src.envs.chess.chess_util import from_fen_to_board, ACTION_NAMES, from_board_to_fen
from src.envs.chess.relace_env import NUM_FEATURES


class ChessGymEnv(gym.Env):

    def __init__(self):
        self.board = chess.Board()
        self.fen = self.board.fen()
        self.state = from_fen_to_board(self.fen)

        lows = np.zeros((2 * NUM_FEATURES + 1,))
        highs = np.ones((2 * NUM_FEATURES + 1,))
        highs.fill(12)
        self.observation_space = gym.spaces.Box(low=lows, high=highs, shape=(2*NUM_FEATURES + 1, ))
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))

    def step(self, move):
        ''' Plays the move, which is in UCI format'''
        try:
            self.board.push(move)
        except ValueError():
            print('Move not found: {}'.format(move))

        self.state = from_fen_to_board(self.board.fen())
        rew = 0

        done = self.board.is_game_over()

        return self.state, rew, done, {}

    def reset(self):
        self.board = chess.Board()
        self.fen = self.board.fen()
        self.state = from_fen_to_board(self.fen)
        return self.state

    def render(self):
        pass

    def close(self):
        pass

    def set_state(self, state):
        new_fen = from_board_to_fen(copy.copy(state))
        self.board = chess.Board(new_fen)
        self.state = copy.copy(state)

        return self.state

    def sample_action(self):
        possible_actions = list(self.board.legal_moves)
        return random.choice(possible_actions)