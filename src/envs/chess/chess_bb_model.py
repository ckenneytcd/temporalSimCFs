import chess
from chess.engine import SimpleEngine
from sklearn.preprocessing import FunctionTransformer
import numpy as np

from src.envs.chess.chess_util import from_board_to_fen, is_valid_fen, is_playable_fen, ACTION_NAMES


class ChessBBModel:
    ''' Wrapper class for Stockfish chess engine '''

    def __init__(self, model_path):
         self.model_path = model_path
         self.engine = self.load_bb_model()
         self.backend = 'sklearn'
         self.transformer = FunctionTransformer(np.log1p)
         self.model_type = 'classifier'

    def predict(self, x):
        ''' Predicts the best move in position in uci format (eg. e2e4) '''
        x = list(x.squeeze())

        fen = from_board_to_fen(x)
        board = chess.Board(fen)

        if is_valid_fen(fen) and is_playable_fen(fen):
         try:
             self.load_bb_model()
             best_move = self.engine.play(board, chess.engine.Limit(time=0.01)).move.uci()
             self.engine.quit()
             return best_move
         except chess.engine.EngineTerminatedError:
             return ''
        else:
         return ''
        return best_move

    def load_bb_model(self):
        """Loads chess engine"""
        engine = SimpleEngine.popen_uci(self.model_path)
        self.engine = engine
        return engine

    def play(self, board):
        return self.engine.play(board, chess.engine.Limit(time=0.01))

    # to be compatible with DICE
    def load_model(self):
        pass

    def get_output(self, cfs):
        ''' Gets the best move for a set of counterfactual instances'''
        # TODO: make sure all algorithms pass cfs as a numpy nd array
        nrows = cfs.shape[0]
        y_hat = np.zeros((nrows, len(ACTION_NAMES)))

        for i in range(nrows):
         cf = cfs[i]

         best_move = self.predict(cf)
         best_move_index = ACTION_NAMES.index(best_move)
         y_hat[i][best_move_index] = 1

        return y_hat

    # for compatibility with DICE
    def get_num_output_nodes2(self, x):
        output = self.get_output(x)
        return output.shape[-1]