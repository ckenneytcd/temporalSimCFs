import json
import pandas as pd

import chess
import numpy as np

from src.envs.chess.chess_bb_model import ChessBBModel
from src.envs.chess.chess_util import from_fen_to_board, from_board_to_fen, ACTION_NAMES, SQUARE_NAMES

from src.envs.chess.chess_util import COL_NAMES


class ChessTask:
    ''' Class that defines the task of finding counterfactual explanations in chess '''

    def __init__(self, model_path, ds_path):
        self.model_path = model_path
        self.ds_path = ds_path
        self.num_piece_types = len(chess.PIECE_TYPES)

        self.bb_model = ChessBBModel(model_path)
        self.dataset = self.create_dataset(100)

    def create_dataset(self, n_ep):
        ''' Creates a dataset of chess positions by having the engine play against itself for n_ep episodes '''
        try:
            # read from file if exists
            dataset = pd.read_csv(self.ds_path, header=0, index_col=0)
            # convert best moves into categorical feature
            dataset['best_move'] = dataset['best_move'].apply(lambda x: ACTION_NAMES.index(x))

            # for sq in SQUARE_NAMES:
            #     dataset[sq] = dataset[sq].apply(str)
            # dataset[SQUARE_NAMES] = dataset[SQUARE_NAMES].astype("category")
            return dataset
        except FileNotFoundError:
            print('File {} not found'.format(self.ds_path))

        if self.bb_model is None:
            raise ValueError('Missing engine. Please load black box model first.')

        # Otherwise generate using engine to play games
        print('Generating dataset... ')
        dataset = []

        # play num_ep games
        for i_ep in range(n_ep):
            board = chess.Board()
            while not board.is_game_over():  # play one game
                result = self.bb_model.play(board)
                board.push(result.move)

                # transform the board into 64 element entry of pieces
                board_entry = from_fen_to_board(board.fen())
                dataset.append(list(board_entry) + [result.move.uci()])

            if i_ep % 100 == 0:
                print('Played {} episodes.'.format(i_ep))

        dataset = np.vstack(dataset)
        dataset = np.unique(dataset, axis=0)  # remove duplicate values
        print('Created dataset with {} positions'.format(dataset.shape[0]))

        # create pd dataframe
        df = pd.DataFrame(dataset, columns=COL_NAMES)
        # store as a csv
        df.to_csv(self.ds_path)
        self.dataset = df
        return df

    def load_factuals(self, json_file):
        ''' Loads chess puzzles from json file '''
        with open(json_file) as f:
            data = json.loads(f.read())

        # extracting fen codes of puzzles
        fen_pos = []
        best_moves = []
        for json_item in data['puzzles']:
            fen_pos.append(json_item['fen'])
            best_moves.append(json_item['best_move'])

        # transforming fen codes into board pieces
        facts = []
        for i, fen in enumerate(fen_pos):
            board = from_fen_to_board(fen)
            facts.append(list(board))

        np_array = np.vstack(facts)
        df = pd.DataFrame(np_array, columns=COL_NAMES[0:65])

        # append the best move column
        df[COL_NAMES[-1]] = best_moves

        # for sq in SQUARE_NAMES + ['player']:
        #     df[sq] = df[sq].apply(str)
        #
        # df[COL_NAMES] = df[COL_NAMES].astype("category")

        return df

    def print_instance(self, board):
        ''' Prints the chess board '''
        fen = from_board_to_fen(board)
        chess_board = chess.Board(fen)

        print(chess_board)




